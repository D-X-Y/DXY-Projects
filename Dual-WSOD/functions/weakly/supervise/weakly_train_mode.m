function mode = weakly_train_mode( conf )
    if ( conf.regression ) 
        if (conf.fast_rcnn)
            mode = 2; %% supervise fast-rcnn train
        else
            mode = 0; %% supervise rfcn train
        end
    else
        mode = 1; %% supervise score only train
    end
end
