import sys

def main():
    with open('src/main.cpp', 'r') as f:
        content = f.read()
        
    start_str = 'const std::string HTML_CONTENT = R"HTML('
    end_str = ')HTML";'
    
    start_idx = content.find(start_str)
    if start_idx == -1:
        print("Could not find HTML_CONTENT")
        return
        
    start_idx_content = start_idx + len(start_str)
    end_idx = content.find(end_str, start_idx_content)
    
    html_content = content[start_idx_content:end_idx]
    
    with open('public/playground.html', 'w') as f:
        f.write(html_content)
        
    # Replace in main.cpp
    # Remove HTML_CONTENT declaration
    new_content = content[:start_idx] + content[end_idx + len(end_str):]
    
    # Replace svr.Get("/") with svr.set_mount_point("/", "./public")
    old_server_code = """        svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(HTML_CONTENT, "text/html");
        });"""
        
    new_server_code = """        auto ret = svr.set_mount_point("/", "./public");
        if (!ret) {
            std::cerr << "The directory './public' doesn't exist.\\n";
        }"""
        
    new_content = new_content.replace(old_server_code, new_server_code)
    
    with open('src/main.cpp', 'w') as f:
        f.write(new_content)
        
    print("Done refactoring main.cpp")

if __name__ == "__main__":
    main()
