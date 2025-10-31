"""Configuration management using Pydantic"""
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class YOLOModel(str, Enum):
    """Available YOLO model sizes"""
    NANO = "yolo11n-seg"
    SMALL = "yolo11s-seg"
    MEDIUM = "yolo11m-seg"
    LARGE = "yolo11l-seg"
    XLARGE = "yolo11x-seg"


class Optimizer(str, Enum):
    """Available optimizers"""
    ADAM = "Adam"
    ADAMW = "AdamW"
    SGD = "SGD"
    RMSPROP = "RMSprop"


class Device(str, Enum):
    """Training device options"""
    CPU = "cpu"
    GPU_0 = "0"
    GPU_1 = "1"
    GPU_ALL = "0,1"


class DatasetFormat(str, Enum):
    """Roboflow export formats"""
    YOLOV11 = "yolov11"
    YOLOV8 = "yolov8"
    YOLOV7 = "yolov7"
    COCO = "coco"


class RoboflowConfig(BaseModel):
    """Roboflow dataset configuration"""
    workspace: str = Field(..., description="Roboflow workspace ID")
    project: str = Field(..., description="Roboflow project ID")
    version: int = Field(..., ge=1, description="Dataset version number")
    format: DatasetFormat = Field(default=DatasetFormat.YOLOV11, description="Export format")
    save_path: Path = Field(default=Path("datasets"), description="Dataset save directory")
    overwrite: bool = Field(default=False, description="Overwrite existing dataset")

    @field_validator("save_path", mode="before")
    @classmethod
    def resolve_path(cls, v):
        return Path(v)

    @property
    def dataset_path(self) -> Path:
        """Get the full dataset path including version"""
        return self.save_path / f"v{self.version}"

    @property
    def data_yaml_path(self) -> Path:
        """Get the path to data.yaml file"""
        return self.dataset_path / "data.yaml"


class TrainingConfig(BaseModel):
    """YOLO training configuration"""
    model_name: YOLOModel = Field(default=YOLOModel.MEDIUM, description="YOLO model size")
    epochs: int = Field(default=300, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Batch size")
    img_size: int = Field(default=640, ge=320, le=1280, description="Image size")
    device: Device = Field(default=Device.GPU_0, description="Training device")
    
    # Optimizer settings
    optimizer: Optimizer = Field(default=Optimizer.ADAMW, description="Optimizer type")
    lr0: float = Field(default=0.01, gt=0, description="Initial learning rate")
    lrf: float = Field(default=0.01, gt=0, description="Final learning rate factor")
    momentum: float = Field(default=0.937, ge=0, le=1, description="SGD momentum/Adam beta1")
    weight_decay: float = Field(default=0.0005, ge=0, description="Weight decay")
    
    # Warmup settings
    warmup_epochs: float = Field(default=3.0, ge=0, description="Warmup epochs")
    warmup_momentum: float = Field(default=0.8, ge=0, le=1, description="Warmup momentum")
    
    # Loss gains
    box_gain: float = Field(default=7.5, ge=0, description="Box loss gain")
    cls_gain: float = Field(default=0.5, ge=0, description="Class loss gain")
    dfl_gain: float = Field(default=1.5, ge=0, description="DFL loss gain")
    
    # Training behavior
    patience: int = Field(default=50, ge=0, description="Early stopping patience")
    save_period: int = Field(default=10, ge=1, description="Save checkpoint every N epochs")
    workers: Optional[int] = Field(default=None, description="Number of dataloader workers")
    cache: bool = Field(default=False, description="Cache images in RAM")
    
    # Output settings
    project_dir: Path = Field(default=Path("models"), description="Project directory")
    pretrain_dir: Path = Field(default=Path("models/pretrain"), description="Pretrained models directory")
    
    # Training features
    amp: bool = Field(default=True, description="Automatic Mixed Precision")
    deterministic: bool = Field(default=True, description="Deterministic mode")
    plots: bool = Field(default=True, description="Generate training plots")
    verbose: bool = Field(default=True, description="Verbose output")
    seed: int = Field(default=0, ge=0, description="Random seed")

    @field_validator("project_dir", "pretrain_dir", mode="before")
    @classmethod
    def resolve_path(cls, v):
        return Path(v)

    @field_validator("workers", mode="before")
    @classmethod
    def set_workers(cls, v):
        if v is None:
            return max(os.cpu_count() // 2, 1)
        return v

    def get_run_name(self, dataset_version: int) -> str:
        """Generate run name for this training"""
        return f"{self.model_name.value}_v{dataset_version}"


class EnvironmentConfig(BaseSettings):
    """Environment variables configuration"""
    roboflow_api_key: str = Field(..., description="Roboflow API key")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class AppConfig(BaseModel):
    """Main application configuration"""
    roboflow: RoboflowConfig
    training: TrainingConfig
    
    @classmethod
    def from_yaml(cls, config_path: str = "src/config/config.yaml") -> "AppConfig":
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_yaml(self, config_path: str = "src/config/config.yaml") -> None:
        """Save configuration to YAML file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and handle enums
        config_dict = self.model_dump(mode="python")
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: str = "src/config/config.yaml") -> tuple[AppConfig, EnvironmentConfig]:
    """Load both application config and environment config"""
    app_config = AppConfig.from_yaml(config_path)
    env_config = EnvironmentConfig()
    return app_config, env_config

def main():
    # Create example configuration
    example_config = AppConfig(
        roboflow=RoboflowConfig(
            workspace="yoloseg-oxj7u",
            project="yolo-seg-okjbk",
            version=4,
            format=DatasetFormat.YOLOV11,
            save_path=Path("datasets"),
            overwrite=False
        ),
        training=TrainingConfig(
            model_name=YOLOModel.MEDIUM,
            epochs=300,
            batch_size=16,
            img_size=640,
            device=Device.GPU_0
        )
    )
    
    # Save example config
    example_config.save_yaml("config.yaml")
    print("✅ Example config.yaml created!")

if __name__ == "__main__":
    main()