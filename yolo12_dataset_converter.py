#!/usr/bin/env python3
"""
YOLO12 Dataset Converter
========================

Convert between different dataset formats for YOLO12 training:
- COCO to YOLO format
- VOC to YOLO format
- YOLO to COCO format
- Dataset validation and statistics

Usage:
    python yolo12_dataset_converter.py --input-format coco --output-format yolo --input-dir /path/to/coco --output-dir /path/to/yolo
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from PIL import Image
import yaml


class DatasetConverter:
    """Convert between different dataset formats"""
    
    def __init__(self):
        self.supported_formats = ['coco', 'yolo', 'voc']
        
    def coco_to_yolo(self, coco_dir: str, output_dir: str) -> bool:
        """Convert COCO format to YOLO format"""
        print(f"Converting COCO to YOLO: {coco_dir} -> {output_dir}")
        
        try:
            # Create output directories
            for split in ['train', 'valid', 'test']:
                os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
            
            # Process each split
            for split in ['train', 'valid', 'test']:
                split_dir = os.path.join(coco_dir, split)
                if not os.path.exists(split_dir):
                    continue
                    
                coco_ann_file = os.path.join(split_dir, '_annotations.coco.json')
                if not os.path.exists(coco_ann_file):
                    print(f"Warning: No COCO annotations found for {split}")
                    continue
                
                # Load COCO data
                with open(coco_ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Create category mapping
                categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
                cat_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
                
                # Create image mapping
                images = {img['id']: img for img in coco_data['images']}
                
                # Process annotations
                annotations_by_image = {}
                for ann in coco_data['annotations']:
                    img_id = ann['image_id']
                    if img_id not in annotations_by_image:
                        annotations_by_image[img_id] = []
                    annotations_by_image[img_id].append(ann)
                
                # Convert each image
                for img_id, img_info in images.items():
                    img_filename = img_info['file_name']
                    img_width = img_info['width']
                    img_height = img_info['height']
                    
                    # Copy image
                    src_img = os.path.join(split_dir, img_filename)
                    dst_img = os.path.join(output_dir, split, 'images', img_filename)
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, dst_img)
                    
                    # Create YOLO labels
                    label_file = os.path.join(output_dir, split, 'labels', 
                                            os.path.splitext(img_filename)[0] + '.txt')
                    
                    with open(label_file, 'w') as f:
                        if img_id in annotations_by_image:
                            for ann in annotations_by_image[img_id]:
                                # Convert COCO bbox to YOLO format
                                x, y, w, h = ann['bbox']
                                x_center = (x + w / 2) / img_width
                                y_center = (y + h / 2) / img_height
                                width = w / img_width
                                height = h / img_height
                                
                                # Get YOLO class ID
                                yolo_class = cat_id_to_yolo[ann['category_id']]
                                
                                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Create dataset YAML
            self.create_yolo_yaml(output_dir, list(categories.values()))
            
            print("✅ COCO to YOLO conversion completed")
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def voc_to_yolo(self, voc_dir: str, output_dir: str) -> bool:
        """Convert VOC format to YOLO format"""
        print(f"Converting VOC to YOLO: {voc_dir} -> {output_dir}")
        
        try:
            # Create output directories
            for split in ['train', 'valid', 'test']:
                os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
                os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
            
            # Collect all classes
            all_classes = set()
            for split in ['train', 'valid', 'test']:
                split_dir = os.path.join(voc_dir, split)
                if not os.path.exists(split_dir):
                    continue
                
                for xml_file in os.listdir(split_dir):
                    if xml_file.endswith('.xml'):
                        tree = ET.parse(os.path.join(split_dir, xml_file))
                        root = tree.getroot()
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            all_classes.add(class_name)
            
            classes = sorted(list(all_classes))
            class_to_id = {cls: idx for idx, cls in enumerate(classes)}
            
            # Process each split
            for split in ['train', 'valid', 'test']:
                split_dir = os.path.join(voc_dir, split)
                if not os.path.exists(split_dir):
                    continue
                
                # Process each XML file
                for xml_file in os.listdir(split_dir):
                    if not xml_file.endswith('.xml'):
                        continue
                    
                    # Parse XML
                    tree = ET.parse(os.path.join(split_dir, xml_file))
                    root = tree.getroot()
                    
                    # Get image info
                    img_width = int(root.find('size/width').text)
                    img_height = int(root.find('size/height').text)
                    img_filename = root.find('filename').text
                    
                    # Copy image
                    src_img = os.path.join(split_dir, img_filename)
                    dst_img = os.path.join(output_dir, split, 'images', img_filename)
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, dst_img)
                    
                    # Create YOLO labels
                    label_file = os.path.join(output_dir, split, 'labels',
                                            os.path.splitext(xml_file)[0] + '.txt')
                    
                    with open(label_file, 'w') as f:
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            if class_name not in class_to_id:
                                continue
                            
                            # Get bounding box
                            bbox = obj.find('bndbox')
                            xmin = float(bbox.find('xmin').text)
                            ymin = float(bbox.find('ymin').text)
                            xmax = float(bbox.find('xmax').text)
                            ymax = float(bbox.find('ymax').text)
                            
                            # Convert to YOLO format
                            x_center = (xmin + xmax) / 2 / img_width
                            y_center = (ymin + ymax) / 2 / img_height
                            width = (xmax - xmin) / img_width
                            height = (ymax - ymin) / img_height
                            
                            yolo_class = class_to_id[class_name]
                            f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Create dataset YAML
            self.create_yolo_yaml(output_dir, classes)
            
            print("✅ VOC to YOLO conversion completed")
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def create_yolo_yaml(self, output_dir: str, classes: List[str]):
        """Create YOLO dataset YAML file"""
        yaml_content = {
            'path': os.path.abspath(output_dir),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(classes),
            'names': classes
        }
        
        yaml_path = os.path.join(output_dir, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ Created dataset YAML: {yaml_path}")
    
    def validate_dataset(self, dataset_dir: str, format_type: str) -> bool:
        """Validate dataset structure and integrity"""
        print(f"Validating {format_type.upper()} dataset: {dataset_dir}")
        
        try:
            if format_type == 'yolo':
                return self.validate_yolo_dataset(dataset_dir)
            elif format_type == 'coco':
                return self.validate_coco_dataset(dataset_dir)
            elif format_type == 'voc':
                return self.validate_voc_dataset(dataset_dir)
            else:
                print(f"❌ Unsupported format: {format_type}")
                return False
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def validate_yolo_dataset(self, dataset_dir: str) -> bool:
        """Validate YOLO dataset"""
        required_dirs = ['train', 'valid']
        
        for split in required_dirs:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                print(f"❌ Missing {split} directory")
                return False
            
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"❌ Missing images or labels directory in {split}")
                return False
            
            # Check for matching files
            image_files = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            label_files = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir) 
                            if f.endswith('.txt'))
            
            if not image_files:
                print(f"❌ No image files found in {split}/images")
                return False
            
            if not label_files:
                print(f"❌ No label files found in {split}/labels")
                return False
            
            # Check for mismatched files
            missing_labels = image_files - label_files
            if missing_labels:
                print(f"⚠️  Warning: {len(missing_labels)} images without labels in {split}")
        
        print("✅ YOLO dataset validation passed")
        return True
    
    def validate_coco_dataset(self, dataset_dir: str) -> bool:
        """Validate COCO dataset"""
        required_dirs = ['train', 'valid']
        
        for split in required_dirs:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                print(f"❌ Missing {split} directory")
                return False
            
            coco_ann = os.path.join(split_dir, '_annotations.coco.json')
            if not os.path.exists(coco_ann):
                print(f"❌ Missing COCO annotations in {split}")
                return False
            
            # Validate COCO JSON
            try:
                with open(coco_ann, 'r') as f:
                    coco_data = json.load(f)
                
                required_keys = ['images', 'annotations', 'categories']
                for key in required_keys:
                    if key not in coco_data:
                        print(f"❌ Missing {key} in COCO annotations")
                        return False
                
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON in {coco_ann}")
                return False
        
        print("✅ COCO dataset validation passed")
        return True
    
    def validate_voc_dataset(self, dataset_dir: str) -> bool:
        """Validate VOC dataset"""
        required_dirs = ['train', 'valid']
        
        for split in required_dirs:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                print(f"❌ Missing {split} directory")
                return False
            
            # Check for XML files
            xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
            if not xml_files:
                print(f"❌ No XML files found in {split}")
                return False
            
            # Validate XML structure
            for xml_file in xml_files[:5]:  # Check first 5 files
                try:
                    tree = ET.parse(os.path.join(split_dir, xml_file))
                    root = tree.getroot()
                    
                    required_elements = ['filename', 'size', 'object']
                    for element in required_elements:
                        if root.find(element) is None:
                            print(f"❌ Missing {element} in {xml_file}")
                            return False
                            
                except ET.ParseError:
                    print(f"❌ Invalid XML in {xml_file}")
                    return False
        
        print("✅ VOC dataset validation passed")
        return True
    
    def get_dataset_stats(self, dataset_dir: str, format_type: str) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'classes': [],
            'splits': {}
        }
        
        try:
            if format_type == 'yolo':
                return self.get_yolo_stats(dataset_dir)
            elif format_type == 'coco':
                return self.get_coco_stats(dataset_dir)
            elif format_type == 'voc':
                return self.get_voc_stats(dataset_dir)
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
        
        return stats
    
    def get_yolo_stats(self, dataset_dir: str) -> Dict:
        """Get YOLO dataset statistics"""
        stats = {'total_images': 0, 'total_annotations': 0, 'classes': set(), 'splits': {}}
        
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
            
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                continue
            
            image_count = len([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            annotation_count = 0
            
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        lines = f.readlines()
                        annotation_count += len(lines)
                        for line in lines:
                            if line.strip():
                                class_id = int(line.split()[0])
                                stats['classes'].add(class_id)
            
            stats['splits'][split] = {
                'images': image_count,
                'annotations': annotation_count
            }
            stats['total_images'] += image_count
            stats['total_annotations'] += annotation_count
        
        stats['classes'] = sorted(list(stats['classes']))
        return stats
    
    def get_coco_stats(self, dataset_dir: str) -> Dict:
        """Get COCO dataset statistics"""
        stats = {'total_images': 0, 'total_annotations': 0, 'classes': [], 'splits': {}}
        
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
            
            coco_ann = os.path.join(split_dir, '_annotations.coco.json')
            if not os.path.exists(coco_ann):
                continue
            
            with open(coco_ann, 'r') as f:
                coco_data = json.load(f)
            
            stats['splits'][split] = {
                'images': len(coco_data['images']),
                'annotations': len(coco_data['annotations'])
            }
            stats['total_images'] += len(coco_data['images'])
            stats['total_annotations'] += len(coco_data['annotations'])
            
            if not stats['classes']:
                stats['classes'] = [cat['name'] for cat in coco_data['categories']]
        
        return stats
    
    def get_voc_stats(self, dataset_dir: str) -> Dict:
        """Get VOC dataset statistics"""
        stats = {'total_images': 0, 'total_annotations': 0, 'classes': set(), 'splits': {}}
        
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
            
            xml_files = [f for f in os.listdir(split_dir) if f.endswith('.xml')]
            annotation_count = 0
            
            for xml_file in xml_files:
                tree = ET.parse(os.path.join(split_dir, xml_file))
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    stats['classes'].add(class_name)
                    annotation_count += 1
            
            stats['splits'][split] = {
                'images': len(xml_files),
                'annotations': annotation_count
            }
            stats['total_images'] += len(xml_files)
            stats['total_annotations'] += annotation_count
        
        stats['classes'] = sorted(list(stats['classes']))
        return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO12 Dataset Converter")
    
    # Conversion arguments
    parser.add_argument("--input-format", type=str, choices=['coco', 'yolo', 'voc'], 
                       help="Input dataset format")
    parser.add_argument("--output-format", type=str, choices=['coco', 'yolo', 'voc'], 
                       help="Output dataset format")
    parser.add_argument("--input-dir", type=str, help="Input dataset directory")
    parser.add_argument("--output-dir", type=str, help="Output dataset directory")
    
    # Validation arguments
    parser.add_argument("--validate", type=str, help="Validate dataset format")
    parser.add_argument("--stats", type=str, help="Get dataset statistics")
    
    args = parser.parse_args()
    
    converter = DatasetConverter()
    
    # Conversion
    if args.input_format and args.output_format and args.input_dir and args.output_dir:
        if args.input_format == 'coco' and args.output_format == 'yolo':
            success = converter.coco_to_yolo(args.input_dir, args.output_dir)
        elif args.input_format == 'voc' and args.output_format == 'yolo':
            success = converter.voc_to_yolo(args.input_dir, args.output_dir)
        else:
            print(f"❌ Conversion from {args.input_format} to {args.output_format} not supported")
            success = False
        
        if success:
            print("✅ Conversion completed successfully!")
        else:
            print("❌ Conversion failed!")
            sys.exit(1)
    
    # Validation
    elif args.validate:
        success = converter.validate_dataset(args.validate, 'yolo')  # Default to YOLO
        if success:
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
            sys.exit(1)
    
    # Statistics
    elif args.stats:
        stats = converter.get_dataset_stats(args.stats, 'yolo')  # Default to YOLO
        print("\n📊 Dataset Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Classes: {stats['classes']}")
        print("\nSplits:")
        for split, data in stats['splits'].items():
            print(f"  {split}: {data['images']} images, {data['annotations']} annotations")
    
    else:
        print("❌ Please specify conversion, validation, or statistics options")
        parser.print_help()


if __name__ == "__main__":
    main()

