import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    # We will split by 'backward_op = ['
    parts = content.split('backward_op = [')
    
    new_content = parts[0]
    
    for part in parts[1:]:
        end_bracket = part.find(']')
        capture_list = part[:end_bracket]
        
        end_lambda = part.find('};')
        body = part[end_bracket:end_lambda]
        
        # Check what we need to replace
        if 'o_impl' in capture_list:
            capture_list = capture_list.replace('o_impl', 'o_ptr = o_impl.get()')
            body = body.replace('o_impl->', 'o_ptr->')
        elif 'loss_impl' in capture_list:
            capture_list = capture_list.replace('loss_impl', 'loss_ptr = loss_impl.get()')
            body = body.replace('loss_impl->', 'loss_ptr->')
        else:
            # Need to match standalone 'o' in capture list. 
            # E.g., 'a, b, o' or 'a, o' or 'a, o, scalar'
            # We can split capture list by comma and check
            caps = [c.strip() for c in capture_list.split(',')]
            new_caps = []
            replaced_o = False
            for c in caps:
                if c == 'o':
                    new_caps.append('o_ptr = o.get()')
                    replaced_o = True
                else:
                    new_caps.append(c)
            
            if replaced_o:
                capture_list = ', '.join(new_caps)
                # Replace 'o->' with 'o_ptr->' in body
                body = body.replace('o->', 'o_ptr->')
                
        new_content += 'backward_op = [' + capture_list + body + part[end_lambda:]

    with open(filepath, 'w') as f:
        f.write(new_content)
        
if __name__ == '__main__':
    fix_file('src/MathEngine.cpp')
    print("Fixed MathEngine.cpp")
