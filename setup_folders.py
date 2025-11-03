"""
Setup script to create necessary output directories for the project
"""
import os

def setup_project_folders():
    """Create all necessary folders for outputs"""
    folders = [
        'outputs/figures',
        'outputs/models',
        'outputs/reports',
        'notebooks'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    print("\n🎉 All folders created successfully!")

if __name__ == "__main__":
    setup_project_folders()
