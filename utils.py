# Complete GPU Memory Release
import gc
import torch

def release_gpu_memory():
    """
    Comprehensive GPU memory cleanup function
    """
    print("🔧 Starting GPU memory cleanup...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        print("   📱 Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        print(f"   📊 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        # Force garbage collection
        print("   🗑️ Running garbage collection...")
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check memory after cleanup
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        
        print(f"   ✅ After cleanup - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        print("   🔄 Peak memory statistics reset")
        
    else:
        print("   ❌ CUDA not available")
    
    # Force Python garbage collection
    collected = gc.collect()
    
    print("✅ GPU memory cleanup completed!")

if __name__=="__main__":
    # Execute cleanup
    release_gpu_memory()

