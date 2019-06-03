function [new_image_roidb_train] = weakly_generate_co_v(train_model, image_roidb_train, pre_keep, SEL_PER_CLS, gamma)

  begin_time = tic;
  caffe.reset_all();
  train_solver = caffe.Solver(train_model.solver_def_file);
  train_solver.net.copy_from( train_model.cur_net_file );
  train_solver.net.set_phase('test');
  classes = train_model.conf.classes;
  number = numel(image_roidb_train);
  Loss = Inf(numel(image_roidb_train), numel(classes));

  for idx = 1:number
    if (rem(idx, 1000) == 0 || idx == number), fprintf('weakly_generate_co_v : handle %4d / %4d image_roidb_train, cost %.2f s\n', idx, number, toc(begin_time)); end

    class = {image_roidb_train(idx).pseudo_boxes.class}; 
    class = cat(1, class{:}); class = unique(class);

    loss = get_loss(train_model.conf, train_solver, image_roidb_train(idx));
    if ( pre_keep(image_roidb_train(idx).index) )
      loss = loss - gamma;
    end
    for j = 1:numel(class)
      Loss(idx, class(j)) = loss;
    end
  end
  cur_keep = false(numel(image_roidb_train), 1);
  for cls = 1:numel(classes)
    [mx_score, mx_ids] = sort(Loss(:, cls));
    %MX_IDS(:, cls) = mx_ids;
    for j = 1:min(number, SEL_PER_CLS(cls))
      if (mx_score(j) < 1)
        cur_keep( mx_ids(j) ) = true;
      else
        break;
      end
    end
  end
  
  new_image_roidb_train = image_roidb_train(cur_keep);
  weakly_debug_info( classes, new_image_roidb_train,  Loss(cur_keep, :));
  fprintf('weakly_generate_co_v %4d -> %4d, cost %.1f s\n', number, numel(new_image_roidb_train), toc(begin_time));
  caffe.reset_all();
end

function loss = get_loss(conf, solver, roidb_train)
  net_inputs = weakly_get_minibatch(conf, roidb_train);
  solver.net.reshape_as_input(net_inputs);
  solver.net.set_input_data(net_inputs);
  solver.net.forward(net_inputs);
  rst = solver.net.get_output();
  if (conf.regression)
      assert (strcmp(rst(2).blob_name, 'loss_bbox') == 1);
      assert (strcmp(rst(3).blob_name, 'loss_cls') == 1);
      loss = rst(2).data + rst(3).data;
  else
      assert (strcmp(rst(2).blob_name, 'loss_cls') == 1);
      loss = rst(2).data;
  end
end
