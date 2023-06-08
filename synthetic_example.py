
if __name__ == '__main__':
    # Hyperparameters
    A_dim, B_dim = 100, 100
    x1_dim, x2_dim = 100, 100
    y_dim = 1
    label_dim = 1
    estimator = 'probabilistic_classifier'
    lr = 1e-4
    relative_ratio = 0.001
    hidden_dim=512 
    embed_dim=128
    layers=1
    activation = 'relu'


    # Define custom dimensions of features and labels
    feature_dim_info = dict()
    label_dim_info = dict()

    intersections = get_intersections(num_modalities=2)

    feature_dim_info['12'] = 10
    feature_dim_info['1'] = 6
    feature_dim_info['2'] = 6

    label_dim_info['12'] = 10
    label_dim_info['1'] = 6
    label_dim_info['2'] = 6

    print(intersections)
    print(feature_dim_info)
    print(label_dim_info)

    # Get datasets
    total_data, total_labels, total_raw_features = generate_data(30000, 2, feature_dim_info, label_dim_info)
    total_labels = get_labels(label_dim_info, total_raw_features)

    dataset = MultimodalDataset(total_data, total_labels)

    # Dataloader
    batch_size = 256
    num_data = total_labels.shape[0]
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*num_data), num_data-int(0.8*num_data)])

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                                batch_size=batch_size,
                                num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                                batch_size=batch_size,
                                num_workers=4)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False,
                                batch_size=batch_size,
                                num_workers=4)
    

    #########################
    #     FactorCL-SUP      #
    #########################

    rus_model = RUSModel(A_dim, B_dim, 20, hidden_dim, embed_dim).cuda()
    train_rusmodel(rus_model, train_loader, dataset, num_epoch=100, num_club_iter=1)
    rus_model.eval()
    
    
