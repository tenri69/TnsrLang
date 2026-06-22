import re

def fix_math_engine():
    with open('src/MathEngine.cpp', 'r') as f:
        content = f.read()
    
    # This regex finds the assignment to backward_op with 'o'
    # e.g., o->backward_op = [a, b, o]() {
    # and replaces it with: o->backward_op = [a, b, o_ptr = o.get()]() {
    
    parts = content.split('backward_op = [')
    new_content = parts[0]
    
    for part in parts[1:]:
        end_bracket = part.find(']')
        capture_list = part[:end_bracket]
        
        # Replace 'o' with 'o_ptr = o.get()' but be careful not to match 'o_impl' or 'o_ptr'
        # Since we know the exact capture lists from the grep, they are:
        # a, b, o
        # a, o, scalar
        # a, o, index
        if ' o]' in capture_list or ', o]' in capture_list or ', o,' in capture_list:
            # We can just replace 'o' as a word
            # Better:
            new_capture_list = re.sub(r'\bo\b', 'o_ptr = o.get()', capture_list)
        else:
            new_capture_list = capture_list
            
        end_lambda = part.find('};')
        body = part[end_bracket:end_lambda]
        
        # Replace o-> with o_ptr-> inside the body
        new_body = body.replace('o->', 'o_ptr->')
        
        new_content += 'backward_op = [' + new_capture_list + new_body + part[end_lambda:]

    # One more thing: what about CrossEntropyLoss?
    # loss_impl->backward_op = [logits_impl, probs, targets, N, C, loss_impl]() {
    new_content = new_content.replace('loss_impl->backward_op = [logits_impl, probs, targets, N, C, loss_impl]', 'loss_impl->backward_op = [logits_impl, probs, targets, N, C, loss_ptr = loss_impl.get()]')
    new_content = new_content.replace('loss_impl->grad', 'loss_ptr->grad')

    with open('src/MathEngine.cpp', 'w') as f:
        f.write(new_content)
    print("Fixed remaining circular references in MathEngine.cpp")

if __name__ == '__main__':
    fix_math_engine()
