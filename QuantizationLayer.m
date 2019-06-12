classdef QuantizationLayer < nnet.layer.Layer
    
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
        function layer = QuantizationLayer(quantizers, codewords)
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
            layer.a = ones(1, codewords);
            layer.b = linspace(-100, 100, codewords);
            layer.c = ones(1, codewords);
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

            tanh_func = @(x) sum(meshgrid(layer.a, x) .* ...
                             tanh(layer.c .* x - meshgrid(layer.b, x)), 2);

            Z = zeros(size(X));
            for ii = 1:size(X, 2)
                for jj = 1:size(X, 3)
                    Z(:, ii, jj) = tanh_func(X(:, ii, jj));
                end
            end
            Z = single(sum(Z, 4));
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

            dZdX_func = @(x) sum(meshgrid(layer.a, x) .* meshgrid(layer.c, x) ./ ...
                                   cosh(layer.c .* x - meshgrid(layer.b, x)).^2, 2);

            dZdb_func = @(x) sum(- meshgrid(layer.a, x) ./ ...
                       cosh(layer.c .* x - meshgrid(layer.b, x)).^2, 1)';

            dLdX = single(zeros(size(X)));
            dLdb = single(zeros(length(layer.b), size(X,2), size(X,3)));
            for ii = 1:size(X, 2)
                for jj = 1:size(X, 3)
                    dLdX(:, ii, jj) = dZdX_func(X(:, ii, jj)) .* dLdZ(:, ii, jj);
                    dLdb(:, ii, jj) = sum(dZdb_func(X(:, ii, jj))' .* dLdZ(:, ii, jj), 1);
                end
            end
            dLdb = sum(sum(dLdb, 2), 3)';
        end
    end
end