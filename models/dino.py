import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment

class NestedTensor:
    def __init__(self, src, mask):
        self.src = src
        self.mask = mask

    def decompose(self):
        return self.src, self.mask

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, tensor_list):
        x = tensor_list.src
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * 3.141592653589793
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * 3.141592653589793

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # 修正済み: 正しい位置エンコーディングの返り値
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos

class DINOv2Backbone(nn.Module):
    def __init__(self, model_size='vits14'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}')
        
        embed_dims = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536
        }
        in_channels = embed_dims[model_size]
        
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(128)

    def forward(self, tensor):
        b, c, h, w = tensor.shape
        
        features = self.backbone.forward_features(tensor)
        patch_tokens = features['x_norm_patchtokens']
        
        h_feat = h // 14
        w_feat = w // 14
        
        # 修正済み: contiguous と reshape を組み合わせて安全に変換
        x = patch_tokens.permute(0, 2, 1).contiguous().reshape(b, -1, h_feat, w_feat)
        out = self.conv(x)
        
        mask = torch.zeros((out.shape[0], out.shape[2], out.shape[3]), dtype=torch.bool, device=out.device)
        nt = NestedTensor(out, mask)
        pos = self.pos_embed(nt)
        
        return [nt], [pos]

class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 1024, 0.1, "relu", batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, 1024, 0.1, "relu", batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.num_decoder_layers = num_decoder_layers

    def forward(self, srcs, masks, poss, query_embed):
        # 修正済み: Transformer専用の安全なメモリ配置
        src = srcs[0].flatten(2).permute(2, 0, 1).contiguous()
        pos = poss[0].flatten(2).permute(2, 0, 1).contiguous()
        
        query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[1], 1)
        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src + pos, src_key_padding_mask=masks[0].flatten(1))
        
        hs = []
        for i in range(self.num_decoder_layers):
            tgt = self.decoder.layers[i](
                tgt + query_embed, memory, 
                memory_key_padding_mask=masks[0].flatten(1)
            )
            # 修正済み: ここもcontiguousで安全に
            hs.append(tgt.permute(1, 0, 2).contiguous())

        return torch.stack(hs)

def build_backbone():
    return DINOv2Backbone('vits14')

def build_transformer():
    return Transformer()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DINONoduleDetector(nn.Module):
    def __init__(self, backbone, transformer, num_classes=1, num_queries=300, d_model=256):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, images):
        features, poss = self.backbone(images)
        srcs = []
        masks = []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)

        hs = self.transformer(srcs, masks, poss, self.query_embed.weight)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if hs.shape[0] > 1:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        return out

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if tgt_bbox.shape[0] == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        out_bbox_xy = box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy')
        tgt_bbox_xy = box_convert(tgt_bbox, in_fmt='cxcywh', out_fmt='xyxy')
        cost_giou = -generalized_box_iou(out_bbox_xy, tgt_bbox_xy)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.reshape(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha=0.25):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss = ce_loss * ((1 - p_t) ** 2.0)
        
        alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
        loss = alpha_t * loss
        return {'loss_ce': loss.mean(1).sum() / num_boxes}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        src_boxes_xy = box_convert(src_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        target_boxes_xy = box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xy, target_boxes_xy))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                l_dict = self.loss_labels(aux_outputs, targets, indices, num_boxes)
                l_dict.update(self.loss_boxes(aux_outputs, targets, indices, num_boxes))
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses