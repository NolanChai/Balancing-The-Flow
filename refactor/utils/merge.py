import os
import csv
import glob

def merge_txt_files_to_csv(source_dir, output_file):
    txt_files = sorted(glob.glob(os.path.join(source_dir, "human_*.txt")))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(['file_name', 'content'])
        
        for txt_file in txt_files:
            file_name = os.path.basename(txt_file)
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                csv_writer.writerow([file_name, content])
                print(f"Processed: {file_name}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    # wrote these in a hurry, modularize later
    source_directory = "/Users/nolan/Documents/GitHub/Balancing-The-Flow/Sources"
    output_csv = "/Users/nolan/Documents/GitHub/Balancing-The-Flow/merged_articles.csv"
    
    merge_txt_files_to_csv(source_directory, output_csv)