import gc
import torch 
import platform
import sys
import psutil
from pathlib import Path
import ultralytics
from setup.config import TrainingConfig, RoboflowConfig
from logger import LOGGER

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def validate_model_file(model_file: Path) -> bool:
    """Validate model file exists and is loadable"""
    
    LOGGER.info(f"Model file path: {model_file}")
    LOGGER.info(f"Model file exists: {model_file.exists()}")
    
    if model_file.exists():
        LOGGER.info(f"Model file size: {model_file.stat().st_size / (1024**2):.2f} MB")
        LOGGER.info("Model file found")
        return True
    else:
        LOGGER.info("Model file not found - will be downloaded")
        return False


def log_system_info():
    """Log detailed system information for debugging"""

    LOGGER.info(f"Python Version: {sys.version}")
    LOGGER.info(f"Platform: {platform.platform()}")
    LOGGER.info(f"Processor: {platform.processor()}")
    
    LOGGER.info(f"CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    LOGGER.info(f"CPU Cores (Logical): {psutil.cpu_count(logical=True)}")
    
    memory = psutil.virtual_memory()
    LOGGER.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    LOGGER.info(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    LOGGER.info(f"RAM Usage: {memory.percent}%")
    
    disk = psutil.disk_usage('/')
    LOGGER.info(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    LOGGER.info(f"Disk Free: {disk.free / (1024**3):.2f} GB")
    LOGGER.info(f"Disk Usage: {disk.percent}%")
    
    try:
        import torch
        LOGGER.info(f"PyTorch Version: {torch.__version__}")
        LOGGER.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"CUDA Version: {torch.version.cuda}")
            LOGGER.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                LOGGER.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                LOGGER.info(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
                LOGGER.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                LOGGER.info(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
    except ImportError:
        LOGGER.warning("PyTorch not available - cannot get GPU info")
    except Exception as e:
        LOGGER.error(f"Error getting GPU info: {e}")
    
    try:
        LOGGER.info(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        LOGGER.warning("Ultralytics not available")

def log_training_config(
        training_config:TrainingConfig, 
        roboflow_config:RoboflowConfig
    ):

    """Log complete training configuration"""
    
    LOGGER.info(f"Model Name: {training_config.model_name.value}")
    LOGGER.info(f"Epochs: {training_config.epochs}")
    LOGGER.info(f"Batch Size: {training_config.batch_size}")
    LOGGER.info(f"Image Size: {training_config.img_size}")
    LOGGER.info(f"Device: {training_config.device.value}")
    LOGGER.info(f"Workers: {training_config.workers}")
    
    LOGGER.info(f"Optimizer: {training_config.optimizer.value}")
    LOGGER.info(f"Learning Rate (lr0): {training_config.lr0}")
    LOGGER.info(f"Final LR Factor (lrf): {training_config.lrf}")
    LOGGER.info(f"Momentum: {training_config.momentum}")
    LOGGER.info(f"Weight Decay: {training_config.weight_decay}")
    
    LOGGER.info(f"Warmup Epochs: {training_config.warmup_epochs}")
    LOGGER.info(f"Warmup Momentum: {training_config.warmup_momentum}")
    
    LOGGER.info(f"Box Gain: {training_config.box_gain}")
    LOGGER.info(f"Class Gain: {training_config.cls_gain}")
    LOGGER.info(f"DFL Gain: {training_config.dfl_gain}")
    
    LOGGER.info(f"Patience: {training_config.patience}")
    LOGGER.info(f"Cache: {training_config.cache}")
    LOGGER.info(f"Plots: {training_config.plots}")
    LOGGER.info(f"Verbose: {training_config.verbose}")
    LOGGER.info(f"Seed: {training_config.seed}")
    LOGGER.info(f"Deterministic: {training_config.deterministic}")
    LOGGER.info(f"AMP: {training_config.amp}")
    
    LOGGER.info(f"Pretrain Dir: {training_config.pretrain_dir.absolute()}")
    LOGGER.info(f"Project Dir: {training_config.project_dir.absolute()}")
    
    LOGGER.info(f"Dataset Version: {roboflow_config.version}")
    LOGGER.info(f"Data YAML Path: {roboflow_config.data_yaml_path}")
    