import re

def fix_math_engine():
    with open('src/MathEngine.cpp', 'r') as f:
        content = f.read()
    
    # We need to replace `o_impl` in the capture list with `o_ptr = o_impl.get()`
    # and inside the lambda body, replace `o_impl->` with `o_ptr->`
    
    # This regex finds the assignment to backward_op
    # e.g., o_impl->backward_op = [a_impl, o_impl]() {
    # and replaces it with: o_impl->backward_op = [a_impl, o_ptr = o_impl.get()]() {
    
    # Strategy: Find all backward_op lambda assignments.
    
    parts = content.split('backward_op = [')
    new_content = parts[0]
    
    for part in parts[1:]:
        # Find the end of the capture list
        end_bracket = part.find(']')
        capture_list = part[:end_bracket]
        
        # Replace o_impl with o_ptr = o_impl.get() in the capture list
        if 'o_impl' in capture_list:
            new_capture_list = capture_list.replace('o_impl', 'o_ptr = o_impl.get()')
        else:
            new_capture_list = capture_list
            
        # The rest of the lambda body is until the next '};'
        end_lambda = part.find('};')
        body = part[end_bracket:end_lambda]
        
        # Replace o_impl-> with o_ptr-> inside the body
        new_body = body.replace('o_impl->', 'o_ptr->')
        
        new_content += 'backward_op = [' + new_capture_list + new_body + part[end_lambda:]

    with open('src/MathEngine.cpp', 'w') as f:
        f.write(new_content)
    print("Fixed circular references in MathEngine.cpp")

if __name__ == '__main__':
    fix_math_engine()
