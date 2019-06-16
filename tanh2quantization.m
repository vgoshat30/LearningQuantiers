function q = tanh2quantization(a, b, c, x)
    % TANH2QUANTIZATION performs a hard quantization on an input scalar.
    % The quantization is defined using a soft, sum of tanh quantization
    % function.
    %
    % Inputs:
    %         x     -   Scalar to quantize
    %         a,b,c -   Parameters of the sum of tanh function, defined by:
    %                   a .* tanh(c .* (x - b))
    % Output:
    %         q     - 	Quantized scalar output
    
    tanh_f = @(x) sum(a .* tanh(c .* (x - b)));
    
    if x <= b(1)
        q = -sum(a);
    elseif x > b(end)
        q = sum(a);
    else
        for codeWord = 2:length(b)
            if b(codeWord-1) < x && x <= b(codeWord)
                middleB = mean(b(codeWord-1:codeWord));
                q = tanh_f(middleB);
                break;
            end
        end
    end
end