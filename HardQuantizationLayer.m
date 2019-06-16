classdef HardQuantizationLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        a
        b
        c
    end
    
    methods
        function layer = HardQuantizationLayer(quantizationLayer)
            % Set layer name.
            layer.Name = 'HardQuantization';
            
            % Set layer description.
            layer.Description = "Hard quantization layer with " ...
                + quantizationLayer.quantizersNum + " quantizers of " ...
                + quantizationLayer.codewordsNum + ...
                " codewords each";
            
            % Set quantization coefs
            layer.a = quantizationLayer.a;
            layer.b = quantizationLayer.b;
            layer.c = quantizationLayer.c;
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
            
            a = layer.a; 
            b = layer.b;
            c = layer.c;
            
            Z = single(zeros(size(X)));
            for jj = 1:size(Z, 2)
                for ii = 1:size(Z, 1)
                    Z(ii,jj) = tanh2quantization(a, b, c, X(ii,jj));
                end
            end
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