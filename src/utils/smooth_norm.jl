"""
    smooth_vector(v::Vector, level::Int=1)

Smooth a vector using a moving average with window size based on smoothing level.

# Arguments
- `v`: Vector to smooth
- `level`: Smoothing intensity (1-5, where 5 is maximum smoothing)

# Returns
- Smoothed vector
"""
function smooth_vector(v::Vector, level::Int=1)
    # Different window sizes for different smoothing levels
    window_sizes = Dict(
        1 => 3,     # Minimal smoothing
        2 => 7,     
        3 => 15,    # Moderate smoothing
        4 => 31,    
        5 => 51     # Maximum smoothing
    )
    
    if !(1 <= level <= 5)
        throw(ArgumentError("Smoothing level must be between 1 and 5"))
    end
    
    window_size = window_sizes[level]
    half_window = div(window_size, 2)
    n = length(v)
    smoothed = similar(v)
    
    # Handle edges and apply smoothing
    for i in 1:n
        start_idx = max(1, i - half_window)
        end_idx = min(n, i + half_window)
        window_vals = v[start_idx:end_idx]
        smoothed[i] = mean(window_vals)
    end
    
    return smoothed
end

"""
    smooth_df(df::DataFrame, level::Int=1; skip_cols::Vector{String}=["time", "freq"])

Smooth all numeric columns in a DataFrame, skipping specified columns.

# Arguments
- `df`: DataFrame to smooth
- `level`: Smoothing intensity (1-5, where 5 is maximum smoothing)
- `skip_cols`: Column names to skip when smoothing

# Returns
- A new DataFrame with smoothed columns
"""
function smooth_df(df::DataFrame, level::Int=1; skip_cols::Vector{String}=["time", "freq"])
    if !(1 <= level <= 5)
        throw(ArgumentError("Smoothing level must be between 1 and 5"))
    end
    
    # Create a new DataFrame to hold the results
    result_df = DataFrame()
    
    # Process each column
    for col_name in names(df)
        col_data = df[!, col_name]
        
        # Skip non-numeric columns and specified skip columns
        if !(eltype(col_data) <: Number) || lowercase(col_name) in lowercase.(skip_cols)
            result_df[!, col_name] = col_data
            continue
        end
        
        # Apply smoothing to numeric columns
        result_df[!, col_name] = smooth_vector(col_data, level)
    end
    
    return result_df
end


"""
    normalize_timeseries(v::Vector; method::Symbol=:zscore, global_params=nothing)

Normalize a vector using either z-score or min-max scaling.

# Arguments
- `v`: Vector to normalize
- `method`: Normalization method (:zscore or :minmax)
- `global_params`: Optional pre-computed parameters for global normalization 
                  (For :zscore, Tuple of (mean, std); For :minmax, Tuple of (min, max))

# Returns
- Normalized vector and the parameters used for normalization
"""
function normalize_timeseries(v::Vector; method::Symbol=:zscore, global_params=nothing)
    if !(method in [:zscore, :minmax])
        throw(ArgumentError("Normalization method must be :zscore or :minmax"))
    end
    
    # Use provided global parameters or compute from the vector
    if global_params === nothing
        if method == :zscore
            μ = mean(v)
            σ = std(v)
            params = (μ, σ)
            
            # Handle constant vectors (prevent division by zero)
            if σ == 0
                return zeros(eltype(v), length(v)), params
            end
            
            normalized = @. (v - μ) / σ
            
        else # :minmax
            min_val = minimum(v)
            max_val = maximum(v)
            params = (min_val, max_val)
            
            # Handle constant vectors
            if min_val == max_val
                return zeros(eltype(v), length(v)), params
            end
            
            normalized = @. (v - min_val) / (max_val - min_val)
        end
    else
        # Use provided global parameters
        if method == :zscore
            μ, σ = global_params
            
            # Handle division by zero
            if σ == 0
                return zeros(eltype(v), length(v)), global_params
            end
            
            normalized = @. (v - μ) / σ
            params = global_params
            
        else # :minmax
            min_val, max_val = global_params
            
            # Handle division by zero
            if min_val == max_val
                return zeros(eltype(v), length(v)), global_params
            end
            
            normalized = @. (v - min_val) / (max_val - min_val)
            params = global_params
        end
    end
    
    return normalized, params
end

"""
    normalize_timeseries_df(df::DataFrame; method::Symbol=:zscore, global_norm::Bool=false,
                 skip_cols::Vector{String}=["time", "freq"])

Normalize all numeric columns in a DataFrame.

# Arguments
- `df`: DataFrame to normalize
- `method`: Normalization method (:zscore or :minmax)
- `global_norm`: If true, normalize all columns using the same parameters
- `skip_cols`: Column names to skip when normalizing

# Returns
- A new DataFrame with normalized columns and the parameters used
"""
function normalize_timeseries_df(df::DataFrame; method::Symbol=:zscore, global_norm::Bool=false,
                      skip_cols::Vector{String}=["time", "freq"])
    
    if !(method in [:zscore, :minmax])
        throw(ArgumentError("Normalization method must be :zscore or :minmax"))
    end
    
    # Create a new DataFrame to hold the results
    result_df = DataFrame()
    norm_params = Dict{String, Any}()
    
    # Identify numeric columns that should be normalized
    norm_cols = [col for col in names(df) if 
                 eltype(df[!, col]) <: Number && 
                 !(lowercase(col) in lowercase.(skip_cols))]
    
    # Calculate global parameters if needed
    global_params = nothing
    if global_norm && !isempty(norm_cols)
        # Combine all values from columns to be normalized
        all_values = vcat([df[!, col] for col in norm_cols]...)
        
        if method == :zscore
            global_params = (mean(all_values), std(all_values))
        else # :minmax
            global_params = (minimum(all_values), maximum(all_values))
        end
    end
    
    # Process each column
    for col_name in names(df)
        col_data = df[!, col_name]
        
        # Skip non-numeric columns and specified skip columns
        if !(eltype(col_data) <: Number) || lowercase(col_name) in lowercase.(skip_cols)
            result_df[!, col_name] = col_data
            continue
        end
        
        # Apply normalization to numeric columns
        normalized_col, params = normalize_vector(col_data; method=method, global_params=global_norm ? global_params : nothing)
        result_df[!, col_name] = normalized_col
        norm_params[col_name] = params
    end
    
    return result_df, norm_params
end

"""
    detrend_vector(v::Vector)

Remove linear trend from a vector by fitting and subtracting a linear regression line.

# Arguments
- `v`: Vector to detrend

# Returns
- Detrended vector
"""
function detrend_vector(v::Vector{T}) where T<:Real
    n = length(v)
    if n <= 1
        return copy(v)
    end
    
    # Create x values (indices)
    x = collect(1:n)
    
    # Calculate linear regression
    # y = mx + b
    x_mean = mean(x)
    y_mean = mean(v)
    
    # Calculate slope (m)
    numerator = sum((x .- x_mean) .* (v .- y_mean))
    denominator = sum((x .- x_mean).^2)
    
    # Handle case where denominator is zero (all x values are the same)
    m = denominator != 0 ? numerator / denominator : 0.0
    
    # Calculate y-intercept (b)
    b = y_mean - m * x_mean
    
    # Create the trend line
    trend = @. m * x + b
    
    # Subtract trend from original data
    detrended = v .- trend
    
    return detrended, trend
end

"""
    detrend_df(df::DataFrame; skip_cols::Vector{String}=["time", "freq"])

Remove linear trend from all numeric columns in a DataFrame, skipping specified columns.

# Arguments
- `df`: DataFrame to detrend
- `skip_cols`: Column names to skip when detrending

# Returns
- A new DataFrame with detrended columns
"""
function detrend_df(df::DataFrame; skip_cols::Vector{String}=["time", "freq"])
    # Create a new DataFrame to hold the results
    result_df = DataFrame()
    trends_df = DataFrame()
    
    # Process each column
    for col_name in names(df)
        col_data = df[!, col_name]
        
        # Skip non-numeric columns and specified skip columns
        if !(eltype(col_data) <: Number) || lowercase(col_name) in lowercase.(skip_cols)
            result_df[!, col_name] = col_data
            trends_df[!, col_name] = zeros(eltype(col_data), length(col_data))  # Placeholder for trend
            continue
        end
        
        # Apply detrending to numeric columns
        detrended_data, trend = detrend_vector(col_data)
        result_df[!, col_name] = detrended_data
        trends_df[!, col_name] = trend
    end
    
    return result_df, trends_df
end


"""
    normalize_spectrum(psd::Vector{T}; eps::Float64=1e-10, log_transform::Bool=true, global_params=nothing)

Normalize a power spectrum by dividing by its sum (or log-sum if log_transform=true).
Optionally use global normalization parameters.

# Arguments
- `psd`: Vector of power spectral density values
- `eps`: Small value to avoid division by zero
- `log_transform`: Whether to apply log10 transform before normalization
- `global_params`: Use provided sum/log-sum for normalization (instead of computing from psd)

# Returns
- Normalized power spectrum (with unit sum)
- The normalization parameter used (sum or log-sum)
"""
function normalize_spectrum(psd::Vector{T}; eps::Float64=1e-10, log_transform::Bool=true, global_params=nothing) where T<:Real
    if log_transform
        psd = log10.(psd .+ eps)
    end
    if global_params === nothing
        total_power = sum(psd) + eps
    else
        total_power = global_params
    end
    return psd ./ total_power, total_power
end

"""
    normalize_spectra(df::DataFrame; freq_col::String="freq", 
                      skip_cols::Vector{String}=["time"], eps=1e-10,
                      log_transform::Bool=true, global_norm::Bool=false)

Normalize power spectrum data in a DataFrame by dividing each column by its sum.
Optionally use global normalization (same sum/log-sum for all columns).

# Arguments
- `df`: DataFrame with spectral data
- `freq_col`: Column name containing frequency values (will be skipped)
- `skip_cols`: Additional column names to skip when normalizing
- `eps`: Small value to avoid division by zero
- `log_transform`: Whether to apply log10 transform before normalization
- `global_norm`: If true, normalize all columns using the same sum/log-sum

# Returns
- A new DataFrame with normalized spectral columns
- A Dict of normalization parameters used for each column (or the global parameter)
"""
function normalize_spectra(df::DataFrame; freq_col::String="freq", 
                              skip_cols::Vector{String}=["time"], eps=1e-10,
                              log_transform::Bool=true, global_norm::Bool=false)
    # Ensure freq_col is in skip_cols if provided
    if freq_col != "" && !(freq_col in skip_cols)
        skip_cols = [skip_cols..., freq_col]
    end

    # Identify columns to normalize
    norm_cols = [col for col in names(df) if 
                 eltype(df[!, col]) <: Number && 
                 !(lowercase(col) in lowercase.(skip_cols))]

    # Compute global normalization parameter if needed
    global_param = nothing
    if global_norm && !isempty(norm_cols)
        # Concatenate all values to a single vector
        all_vals = vcat([df[!, col] for col in norm_cols]...)
        if log_transform
            all_vals = log10.(all_vals .+ eps)
        end
        global_param = sum(all_vals) + eps
    end

    # Create a new DataFrame to hold the results
    result_df = DataFrame()
    norm_params = Dict{String, Any}()

    # Add frequency column as is
    if freq_col != "" && freq_col in names(df)
        result_df[!, freq_col] = df[!, freq_col]
    end

    # Process each column
    for col_name in names(df)
        # Skip non-numeric columns and specified skip columns
        if lowercase(col_name) in lowercase.(skip_cols)
            if !(col_name == freq_col && freq_col in names(result_df))
                result_df[!, col_name] = df[!, col_name]
            end
            continue
        end

        col_data = df[!, col_name]
        if !(eltype(col_data) <: Number)
            result_df[!, col_name] = col_data
            continue
        end

        # Normalize the spectrum (divide by sum)
        normed, param = normalize_spectrum(col_data; eps=eps, log_transform=log_transform, global_params=global_norm ? global_param : nothing)
        result_df[!, col_name] = normed
        norm_params[col_name] = param
    end

    return result_df
end