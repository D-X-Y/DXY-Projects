function model_path = weakly_supervised_score(roidb_train, solver_file, model_file, val_interval, conf, cache_dir, prefix, suffix, final_name)
  model_path = fullfile(cache_dir, [prefix, '_', final_name, suffix]);
  %if (exist(model_path, 'file'))
  %  fprintf('Exist Caffe Model : %s, Skiped\n', model_path);
  %  return;
  %end
  caffe.reset_all();
  caffe_solver = caffe.Solver(solver_file);
  caffe_solver.net.copy_from(model_file);
  total_num = numel(roidb_train);
  caffe_solver.set_max_iter(total_num * conf.max_epoch);
  caffe_solver.set_stepsize(total_num * conf.step_epoch);

  shuffled_inds = [];
  train_results = [];
  max_iter = caffe_solver.max_iter();
  fprintf('********** %6s Score Training : total[%5d] max_iter[%6d] *************\n', prefix, total_num, max_iter);
  caffe_solver.net.set_phase('train');

  while (caffe_solver.iter() < max_iter)

    [shuffled_inds, sub_db_inds] = weakly_generate_random_minibatch(shuffled_inds, roidb_train, conf.ims_per_batch);
    net_inputs = weakly_get_minibatch(conf, roidb_train(sub_db_inds));

    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);

    rst = caffe_solver.net.get_output();
    train_results = parse_rst(train_results, rst);

    % do valdiation per val_interval iterations
    if mod(caffe_solver.iter(), val_interval) == 0
        weakly_show_state(caffe_solver.iter(), max_iter, train_results);
        train_results = [];
        diary; diary; % flush diary
    end
    % weakly_snapshot
    if mod(caffe_solver.iter(), total_num) == 0
        iter_model_path = fullfile(cache_dir, [prefix, '_epoch_', num2str(caffe_solver.iter()/total_num), suffix]);
        caffe_solver.net.save(iter_model_path);
        fprintf('weakly_supervised_score only save model : %s\n', iter_model_path);
    end

  end
  % final weakly_snapshot
  caffe_solver.net.save(model_path);
  fprintf('weakly_supervised_score only save model : %s\n', model_path);
  caffe.reset_all();
end

function [shuffled_inds, sub_inds] = weakly_generate_random_minibatch(shuffled_inds, image_roidb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory

        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);

        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));

        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);

        shuffled_inds = [hori_image_inds, vert_image_inds];
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));

        shuffled_inds = num2cell(shuffled_inds, 1);
    end

    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function weakly_show_state(iter, max_iter, train_results)
    fprintf('------------------------ %10s Iteration %4d / %4d -------------------------\n', datestr(datevec(now()), '[yyyy:mm:dd]@[HH:MM:SS]'), iter, max_iter);
    fprintf('Training : accuracy %.3g, loss (cls %.3g)\n', ...
        mean(train_results.accuracy.data), ...
        mean(train_results.loss_cls.data));
end
