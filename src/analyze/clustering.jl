using Clustering
using Statistics

"""
    silhouette_from_D(D, labels) -> Vector{Float64}

Compute per-item silhouette scores from a square pairwise distance matrix.
Singleton clusters receive score 0 by convention.
"""
function silhouette_from_D(D::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer})
    n = size(D, 1)
    K = maximum(labels)
    s = zeros(Float64, n)

    for i in 1:n
        ci = labels[i]

        # indices in same cluster (excluding i)
        same = [j for j in 1:n if j != i && labels[j] == ci]
        if isempty(same)
            # cluster of size 1 → silhouette is 0 by convention
            s[i] = 0.0
            continue
        end

        # a(i): mean intra–cluster distance
        a = mean(D[i, same])

        # b(i): smallest mean distance to any other cluster
        b = Inf
        for c in 1:K
            c == ci && continue
            inds = [j for j in 1:n if labels[j] == c]
            isempty(inds) && continue
            dmean = mean(D[i, inds])
            if dmean < b
                b = dmean
            end
        end

        s[i] = (b - a) / max(a, b)
    end

    return s
end

function silhouette_curve(D::Matrix; k_range = 2:10)
    
    hc = hclust(D, linkage=:average)
    ks = collect(k_range)
    scores = Float64[]

    for k in ks
        labels = cutree(hc;k=k)
        s = silhouette_from_D(D, labels)
        push!(scores, mean(s))
    end

    return ks, scores
end

function best_k_clustering(D::AbstractMatrix; k_range = 2:7, linkage = :average)
    # D is n×n symmetric distance matrix
    best_k      = nothing
    best_labels = nothing
    best_score  = -Inf

    hc = hclust(D, linkage = linkage, branchorder = :optimal)

    for k in k_range
        labels = cutree(hc;k=k)
        # silhouettes has a method that accepts a distance matrix directly
        s      = silhouette_from_D(D, labels)
        score  = mean(s)

        # keep the best k
        if score > best_score
            best_score  = score
            best_k      = k
            best_labels = labels
        end
    end

    return best_k, best_labels, best_score
end
