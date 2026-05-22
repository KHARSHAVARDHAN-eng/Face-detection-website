import os
import sys

# Add backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

try:
    import torch
    from models.cdcn_model import CDCN
except ImportError:
    print("\n[CDCN EXPORT] ERROR: PyTorch ('torch') is not installed in the current environment.")
    print("To run the conversion, please install PyTorch in the virtual environment:")
    print("  ./venv/bin/python -m pip install torch\n")
    sys.exit(1)

def main():
    print("Instantiating CDCN model...")
    model = CDCN(theta=0.7)
    model.eval()
    
    # CDCN inputs are cropped face images of size 256x256
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    
    output_path = os.path.join(backend_dir, "models", "cdcn.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting CDCN model to ONNX format: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_face"],
        output_names=["depth_map"],
        dynamic_axes={
            "input_face": {0: "batch_size"},
            "depth_map": {0: "batch_size"}
        }
    )
    print("[CDCN EXPORT] Export completed successfully!")
    print(f"[CDCN EXPORT] Saved ONNX model to: {output_path}")

if __name__ == "__main__":
    main()
