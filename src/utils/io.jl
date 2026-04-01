function load_data(ds::DataSettings)
    # Validate inputs
    data_file = joinpath(ds.data_path, ds.data_fname)
    data = CSV.read(data_file, DataFrame)
    return data
end