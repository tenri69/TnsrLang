import os
import datetime

AUTHOR = "tenri69"
DATE = "2026-06-20"

HEADER_TEMPLATE = """/*
 * =====================================================================================
 *
 *       Filename:  {filename}
 *
 *    Description:  TnsrLang Interpreter Engine Component
 *
 *        Version:  1.0
 *        Created:  {date}
 *       Revision:  none
 *       Compiler:  clang++
 *
 *         Author:  {author}
 *   Organization:  TnsrLang Open Source
 *
 * =====================================================================================
 */

"""

def add_header(filepath):
    if "httplib.h" in filepath:
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if header already exists
    if "Author:  tenri69" in content:
        return
        
    filename = os.path.basename(filepath)
    header = HEADER_TEMPLATE.format(filename=filename, date=DATE, author=AUTHOR)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.h'):
                add_header(os.path.join(root, file))

if __name__ == "__main__":
    process_directory("src")
    print("Headers added successfully.")
