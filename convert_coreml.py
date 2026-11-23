"""Convert pretrained ImageNet timm weights, separately from CIFAR-10 checkpoints."""
import argparse
import json
from pathlib import Path

import coremltools as ct
import timm
import torch
from torchvision.models import ResNet18_Weights


class ImageClassifier(torch.nn.Module):
    """Normalize each RGB channel exactly; ImageType first scales pixels to [0, 1]."""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        return self.model((image - self.mean) / self.std).softmax(dim=1)


def convert_model_to_coreml(model_name, output_dir="build/coreml", labels_path=None):
    output = Path(output_dir) / f"{model_name}.mlpackage"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose another --output-dir")
    model = timm.create_model(model_name, pretrained=True, scriptable=False, exportable=True).eval()
    config = timm.data.resolve_model_data_config(model)
    # These are the standard 1000 ImageNet categories in torchvision logit order.
    labels = (Path(labels_path).read_text().splitlines() if labels_path else
              ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
    if len(labels) != model.num_classes or any(not label.strip() for label in labels):
        raise ValueError(f"Expected {model.num_classes} nonempty labels in classifier logit order")
    shape = (1,) + tuple(config["input_size"])
    wrapped = ImageClassifier(model, config["mean"], config["std"]).eval()
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, torch.rand(shape))
    converted = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=shape, scale=1 / 255.0)],
        classifier_config=ct.ClassifierConfig(labels), skip_model_load=True,
        minimum_deployment_target=ct.target.iOS15,
    )
    converted.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageClassifier"
    converted.user_defined_metadata["timm_model"] = model_name
    converted.user_defined_metadata["preprocessing"] = json.dumps(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))
    (output.parent / f"{model_name}.preprocessing.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {output}. Resize/crop input according to the preprocessing sidecar.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Use an explicit pretrained timm weight tag")
    parser.add_argument("--output-dir", default="build/coreml")
    parser.add_argument("--labels", help="Optional UTF-8 label file, one class per logit; default: torchvision ImageNet categories")
    args = parser.parse_args()
    convert_model_to_coreml(args.model, args.output_dir, args.labels)


if __name__ == "__main__":
    main()
