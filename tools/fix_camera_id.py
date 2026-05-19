import os
import re

def fix_camera_ids(data_dir='data/processed/gzgc_zebra'):
    """
    - query: c1 -> c2
    - gallery: c1 -> c3
    """
    
    query_dir = os.path.join(data_dir, 'query')
    if os.path.exists(query_dir):
        count = 0
        for pid_folder in os.listdir(query_dir):
            pid_path = os.path.join(query_dir, pid_folder)
            if not os.path.isdir(pid_path):
                continue
            for fname in os.listdir(pid_path):
                if '_c1_' in fname:
                    old_path = os.path.join(pid_path, fname)
                    new_fname = fname.replace('_c1_', '_c2_')
                    new_path = os.path.join(pid_path, new_fname)
                    os.rename(old_path, new_path)
                    count += 1
        print(f"Query: repaired {count} files")
    
    gallery_dir = os.path.join(data_dir, 'gallery')
    if os.path.exists(gallery_dir):
        count = 0
        for pid_folder in os.listdir(gallery_dir):
            pid_path = os.path.join(gallery_dir, pid_folder)
            if not os.path.isdir(pid_path):
                continue
            for fname in os.listdir(pid_path):
                if '_c1_' in fname:
                    old_path = os.path.join(pid_path, fname)
                    new_fname = fname.replace('_c1_', '_c3_')
                    new_path = os.path.join(pid_path, new_fname)
                    os.rename(old_path, new_path)
                    count += 1
        print(f"Gallery: repaired {count} files")
    
    print("Done!")

if __name__ == '__main__':
    fix_camera_ids()
