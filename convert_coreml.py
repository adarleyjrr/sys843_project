import coremltools as ct
import timm
import torch
import statistics
import argparse

# model_name = "fastvit_t8.apple_in1k" # 4.05M
# model_name = "fastvit_sa12.apple_in1k"  # 11.6M
# model_name = "mobileone_s1.apple_in1k" # 4.89M
# model_name = "mobileone_s3.apple_in1k"  # 10.3M
# model_name = "tiny_vit_11m_224.dist_in22k_ft_in1k" # 11M
# model_name = "tiny_vit_21m_224.dist_in22k_ft_in1k"  # 21.2M
# model_name = "resnet18.a1_in1k" # 11.7M
model_name = "resnet34.a1_in1k"  # 21.8M


def convert_model_to_coreml(model_name):
    print(f"Creating model {model_name}")
    timm_model = timm.create_model(
        model_name, pretrained=True, scriptable=False, exportable=True
    )

    model = torch.nn.Sequential(timm_model, torch.nn.Softmax(1)).eval()

    input_size = timm_model.default_cfg.get("input_size")
    input_shape = (1,) + input_size

    print("Tracing model")
    example_input = torch.randn(input_shape)
    jit_model = torch.jit.trace(model, example_input)

    labels_filename = "imagenet-classes.txt"

    with open(labels_filename, "r") as labels_file:
        labels = [line.strip() for line in labels_file.readlines()]

    classifier_config = ct.ClassifierConfig(labels)

    print("Converting model")
    mean = list(timm_model.default_cfg.get("mean"))
    std = list(timm_model.default_cfg.get("std"))

    mean_std = statistics.mean(std)
    scale = 1 / (mean_std * 255)
    bias = [-m / s for m, s in zip(mean, std)]
    input_type = ct.ImageType(name="image", shape=input_shape, scale=scale, bias=bias)

    coreml_model = ct.convert(
        jit_model,
        convert_to="mlprogram",
        inputs=[input_type],
        classifier_config=classifier_config,
        skip_model_load=True,
    )

    coreml_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = (
        "imageClassifier"
    )

    coreml_model_file_name = f"coreml_models/{model_name}.mlpackage"
    print(f"Saving model to {coreml_model_file_name}")

    coreml_model.save(coreml_model_file_name)
    print("Done!")


def main():

    parser = argparse.ArgumentParser(description="Convert TIMM model to CoreML")
    parser.add_argument("--model", type=str, required=True, help="Model name from TIMM")
    args = parser.parse_args()

    convert_model_to_coreml(args.model)


if __name__ == "__main__":
    main()
