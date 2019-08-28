classdef QuantizationLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        quantizersNum
        codewordsNum
        a
        c
    end
    
    properties (Learnable)
        % Layer learnable parameters
        b
    end
    
    methods
        function layer = QuantizationLayer(quantizers, codewords, max_in, max_out)
            % Set number of inputs.
            layer.quantizersNum = quantizers;
            layer.codewordsNum = codewords;
            
            % Set layer name.
            layer.Name = 'Quantization';
            
            % Set layer description.
            layer.Description = "Learning quantization layer with " ...
                + quantizers + " quantizers of " + codewords + ...
                " codewords each";
            
            % Initialize layer weights
            layer.a = max_out/codewords * ones(1, codewords-1);
            % FIXME: b and c should not be multiplied
            layer.b = max_in*linspace(-1, 1, codewords-1);
            layer.c = 12/mean(diff(layer.b)) * ones(1, codewords-1);
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
                    Z(ii,jj) = sum(a .* tanh(c .* (X(ii,jj) - b)));
                end
            end
        end
        
        function [dLdX, dLdb] = backward(layer, X, ~, dLdZ, ~)
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

            a = layer.a; 
            b = layer.b;
            c = layer.c;
            
            dLdX = single(zeros(size(X)));
            for jj = 1:size(dLdX, 2)
                for ii = 1:size(dLdX, 1)
                    dZidXi = sum(a .* c .* (1-(tanh(c .* (X(ii,jj) - b))).^2));
                    dLdX(ii,jj) = dLdZ(ii,jj) * dZidXi;
                end
            end
            
            dLdb = single(zeros(size(layer.b)));
            for jj = 1:size(dLdX, 2)
                for ii = 1:size(dLdb, 2)
                    dZdbi_in = (tanh(c(ii)* (X(:,jj) - b(ii)))).^2 - 1;
                    dLdb(ii) = a(ii) * c(ii) * sum(dLdZ(:,jj) .* dZdbi_in);
                end
            end
        end
    end
end