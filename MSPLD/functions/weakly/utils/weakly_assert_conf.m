function weakly_assert_conf(confs)
    num = numel(confs);
    for j = 2:num
        assert (confs{j}.use_flipped == confs{j-1}.use_flipped);
        assert (confs{j}.use_gpu == confs{j-1}.use_gpu);
        %assert (confs{j}.bbox_class_agnostic == confs{j-1}.bbox_class_agnostic);
        assert (confs{j}.test_binary == confs{j-1}.test_binary);
    end
    for j = 1:num
        assert (isfield(confs{j}, 'classes'));
        assert (isfield(confs{j}, 'max_rois_num_in_gpu'));
    end
end
