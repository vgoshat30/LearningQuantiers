classdef SamplingLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        sigma
        T
        Ttilde
        p
    end
    
    properties (Learnable)
        % Layer learnable parameters
        phi
    end
    
    methods
        function layer = SamplingLayer(samplers, observedT, smaplesNum)
            layer.T = observedT;
            layer.p = samplers;
            layer.Ttilde = smaplesNum;
            
            % Set layer name.
            layer.Name = 'Sampling';
            
            % Set layer description.
            layer.Description = "Learning sampling layer with " ...
                + samplers + " samplers of " + smaplesNum + ...
                " samples each";
            
            % Initialize layer weights
            samplTimesSpread = linspace(0, observedT, smaplesNum+2);
            samplTimesSpread(1) = [];
            samplTimesSpread(end) = [];
            layer.phi = ceil(samplTimesSpread);
            layer.sigma = 0.4 * ones(1, smaplesNum);
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
            sigma = layer.sigma;
            Tt = layer.Ttilde;
            T = layer.T;
            p = layer.p;
            t = 1:T;
            
            Z = single(zeros(Tt * p, size(X,2)));
            for jj = 1:size(Z, 2)
                for kk = 1:p
                    for ii = 1:Tt
                        Z((kk-1)*Tt+ii,jj) = sum(X((kk-1)*T+t,jj) .* ...
                                                 exp(-((t-phi(ii))./sigma(ii)).^2)');
                    end
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

            % Write parameters conviniently
            phi = layer.phi; 
            sigma = layer.sigma;
            Tt = layer.Ttilde;
            T = layer.T;
            p = layer.p;
            ii = 1:Tt;
            
            dLdX = single(zeros(size(X)));
            for jj = 1:size(dLdX, 2)
                for kk = 1:p
                    for t = 1:T
                        dZidXi = exp(-((t-phi)./sigma).^2)';
                        dLdX((kk-1)*T+t,jj) = sum(dLdZ((kk-1)*Tt+ii,jj).*dZidXi);
                    end
                end
            end
            
            t = 1:T;
            
            dLdb = single(zeros(size(phi)));
            for jj = 1:size(dLdX, 2)
                for ii = 1:Tt
                    dZdphi = zeros(p, 1);
                    for kk = 1:p
                        dZdphi(kk) = 2*sum(X((kk-1)*T+t,jj) .* (t'-phi(ii))./sigma(ii) .* ...
                                                     exp(-((t-phi(ii))./sigma(ii)).^2)');
                    end
                    kk = 1:p;
                    dLdb(ii) = sum(dLdZ((kk-1)*Tt+ii,jj) .* dZdphi);
                end
            end
        end
    end
end