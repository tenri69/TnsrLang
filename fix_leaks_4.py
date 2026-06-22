import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    parts = content.split('backward_op = [')
    
    new_content = parts[0]
    
    for part in parts[1:]:
        end_bracket = part.find(']')
        capture_list = part[:end_bracket]
        end_lambda = part.find('};')
        body = part[end_bracket:end_lambda]
        
        # Match standalone 'o' in capture list. 
        caps = [c.strip() for c in capture_list.split(',')]
        new_caps = []
        replaced_o = False
        for c in caps:
            if c == 'o':
                new_caps.append('o_ptr = o.get()')
                replaced_o = True
            elif c == 'loss_impl':
                new_caps.append('loss_ptr = loss_impl.get()')
            else:
                new_caps.append(c)
        
        if replaced_o:
            body = body.replace('o->', 'o_ptr->')
        
        if 'loss_impl' in capture_list:
            body = body.replace('loss_impl->', 'loss_ptr->')
            
        capture_list = ', '.join(new_caps)
        new_content += 'backward_op = [' + capture_list + body + part[end_lambda:]

    with open(filepath, 'w') as f:
        f.write(new_content)
        
if __name__ == '__main__':
    fix_file('src/MathEngine.cpp')
