import matplotlib.pyplot as plt
from torchinfo import summary
from Networks.network import ObjectDetectionModel

model = ObjectDetectionModel(num_classes=3, feature_dim=64)
result = summary(model, input_size=(1, 5000, 3), depth=4,
                 col_names=["input_size", "output_size", "num_params", "trainable"],
                 verbose=0)
summary_str = str(result)

fig, ax = plt.subplots(figsize=(16, len(summary_str.splitlines()) * 0.22 + 1))
ax.axis("off")
ax.text(0, 1, summary_str, transform=ax.transAxes, fontsize=7,
        verticalalignment="top", fontfamily="monospace")
plt.tight_layout()
plt.savefig("model_summary.png", dpi=200, bbox_inches="tight")
print("Saved model_summary.png")
