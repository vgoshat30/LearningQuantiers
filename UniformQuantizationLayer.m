classdef UniformQuantizationLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        codewordsNum
        support
    end
    
    methods
        function layer = UniformQuantizationLayer(codewordsNum, support)
            % Set layer name.
            layer.Name = 'UniformQuantization';
            
            % Set layer description.
            layer.Description = "Uniform quantization layer with " ...             
                + codewordsNum + ...
                " codewords";
            
            % Set quantization coefs
            layer.support = support;
            layer.codewordsNum = codewordsNum;
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
                        
            
            Z = m_fQuant(X, layer.codewordsNum, layer.support);
        end
        
        function dLdX = backward(~, X, ~, ~, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation [unused]
            % Output:
            %         dLdX              - Derivative of the loss with
            %                             respect to the input data
            %         dLdAlpha          - Derivatives of the loss with
            %                             respect to alpha
            
            dLdX = single(zeros(size(X)));
        end
    end
end