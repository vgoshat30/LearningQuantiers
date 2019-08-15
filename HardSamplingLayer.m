classdef HardSamplingLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        sigma
        T
        Ttilde
        p
        phi
    end
    
    methods
        function layer = HardSamplingLayer(SamplingLayer)
            % Set layer name.
            layer.Name = 'HardSampling';
            
            layer.T = SamplingLayer.T;
            layer.p = SamplingLayer.p;
            layer.Ttilde = SamplingLayer.Ttilde;
            layer.sigma = SamplingLayer.sigma;
            
            % Set layer description.
            layer.Description = "Hard sampling layer with " ...
                + layer.p + " samplers of " + layer.Ttilde + ...
                " samples each";
          
            % Restrain gaussian shifts to T
            phi = round(SamplingLayer.phi);
            phi(phi < 1) = 1;
            phi(phi > layer.T) = layer.T;
            layer.phi = phi;
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
            
            % Write parameters conviniently
            phi = layer.phi;
            Tt = layer.Ttilde;
            T = layer.T;
            p = layer.p;
            
            Z = single(zeros(Tt * p, size(X,2)));
            for jj = 1:size(Z, 2)
                for kk = 1:p
                    for ii = 1:Tt
                        Z((kk-1)*Tt+ii,jj) = X((kk-1)*T+phi(ii),jj);
                    end
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