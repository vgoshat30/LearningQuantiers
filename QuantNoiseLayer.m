classdef QuantNoiseLayer < nnet.layer.Layer
    % QuantNoiseLayer   Quantizantion noise layer
    %   A quantization noise layer adds random uniforom noise to the input.
    %
    %   To create a quantization noise layer, use 
    %   layer = QuantNoiseLayer(sigma)

    properties
        % Support.
        Sigma
    end
    
    methods
        function layer = QuantNoiseLayer(sigma)
            % layer = QuantNoiseLayer(sigma,name) creates a Quant
            % noise layer and specifies the standard deviation and layer
            % name.
            
            layer.Name = 'Quantization noise';
            layer.Description = ...
                "Quantization noise with support " + sigma;
            layer.Type = "Quanization noise";
            layer.Sigma = sigma;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer for prediction and outputs the result Z.
            
            % At prediction time, the output is equal to the input.
            Z = X;
        end
        
        function [Z, memory] = forward(layer, X)
            % Z = forward(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            % At training time, the layer adds uniform noise to the input.
            sigma = layer.Sigma;
            noise = (rand(size(X)) - 0.5) * (sigma/2);
            Z = X + noise;
            
            memory = [];
        end
        
        function dLdX = backward(layer, X, Z, dLdZ, memory)
            % [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            % Since the layer adds a random constant, the derivative dLdX
            % is equal to dLdZ.
            dLdX = dLdZ;
        end
    end
end