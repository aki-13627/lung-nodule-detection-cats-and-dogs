import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer

def main():
    checkpoint = "SenseTime/deformable-detr"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    
    id2label = {0: "nodule"}
    label2id = {"nodule": 0}
    
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir="deformable_detr_nodules",
        per_device_train_batch_size=4,
        num_train_epochs=100,
        fp16=True,
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == '__main__':
    main()