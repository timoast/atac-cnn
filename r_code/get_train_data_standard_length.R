library(GenomicRanges)
library(Signac)
library(Seurat)
library(Matrix)

pbmc <- readRDS("data/pbmc_multiomic.rds")

# remove low count cell types
cell_counts <- table(Idents(pbmc))
pbmc <- pbmc[, pbmc$celltype %in% names(cell_counts[cell_counts > 50])]

# find midpoint of each peak, resize to 1 kb, quantify and make new assay
peaks <- granges(pbmc)

# resize
peaks_1k <- resize(peaks, width = 1000, fix = "center")

counts_1k <- FeatureMatrix(
  fragments = Fragments(pbmc),
  features = peaks_1k,
  cells = colnames(pbmc)
)

pbmc[["onek"]] <- CreateChromatinAssay(counts = counts_1k)

normalize_counts <- function(object, assay = "ATAC", group.by = "celltype") {
  counts <- GetAssayData(object = object, assay = assay, slot = "counts")
  
  # get scale factors
  reads.per.group <- AverageCounts(
    object = object,
    group.by = group.by,
    verbose = FALSE
  )
  cells.per.group <- CellsPerGroup(
    object = object,
    group.by = group.by
  )
  scale.factors <- reads.per.group * cells.per.group
  
  # get totals
  ident.matrix <- Signac:::BinaryIdentMatrix(object = object, group.by = group.by)
  collapsed.counts <- tcrossprod(x = counts, y = ident.matrix)
  
  # divide by scale factors to account for cell number and depth
  collapsed.counts.scaled <- tcrossprod(collapsed.counts, Diagonal(x = 1/scale.factors[colnames(collapsed.counts)]))
  
  # multiply by constant
  collapsed.counts.scaled <- collapsed.counts.scaled * 1e6
  return(as.matrix(collapsed.counts.scaled))
}


# normalize_counts <- function(object, assay = "ATAC", group.by = "celltype") {
#   counts <- GetAssayData(object = object, assay = assay, slot = "counts")
#   
#   # get widths
#   widths <- width(StringToGRanges(rownames(counts)))
#   
#   # get scale factors
#   reads.per.group <- AverageCounts(
#     object = object,
#     group.by = group.by,
#     verbose = FALSE
#   )
#   cells.per.group <- CellsPerGroup(
#     object = object,
#     group.by = group.by
#   )
#   scale.factors <- reads.per.group * cells.per.group
#   
#   # get totals
#   ident.matrix <- Signac:::BinaryIdentMatrix(object = object, group.by = group.by)
#   collapsed.counts <- tcrossprod(x = counts, y = ident.matrix)
#   
#   # divide by width
#   collapsed.counts.norm <- t(crossprod(collapsed.counts, Diagonal(x = 1/widths)))
#   
#   # divide by scale factors to account for cell number and depth
#   collapsed.counts.scaled <- tcrossprod(collapsed.counts.norm, Diagonal(x = 1/scale.factors[colnames(collapsed.counts.norm)]))
#   
#   # multiply by constant
#   collapsed.counts.scaled <- collapsed.counts.scaled * 1e8
#   return(as.matrix(collapsed.counts.scaled))
# }

avg.counts <- normalize_counts(object = pbmc, assay = "onek")

CoveragePlot(pbmc, region = "chr12-119988742-119989741", extend.upstream = 5000, extend.downstream = 5000, ranges = granges(pbmc[['onek']]))

# write matrix to file

# randomize order
avg.counts <- avg.counts[sample(rownames(avg.counts), size = nrow(avg.counts), replace = FALSE), ]

# split into train, test
n_examples <- nrow(avg.counts)
train_idx <- rownames(avg.counts)[1:(0.8*n_examples)]
test_idx <- setdiff(rownames(avg.counts), train_idx)
write.table(x = avg.counts[train_idx, ], file = "data/training_data_regression_1kb.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = avg.counts[test_idx, ], file = "data/test_data_regression_1kb.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")

# convert to z-score
centered <- avg.counts - rowMeans(avg.counts)
scaled <- centered / apply(avg.counts, 1, sd)

write.table(x = scaled[train_idx, ], file = "data/training_data_regression_1kb_zscore.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = scaled[test_idx, ], file = "data/test_data_regression_1kb_zscore.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")

# convert to relative score
maxvals <- apply(avg.counts, 1, max)
relative_access <- avg.counts / maxvals

write.table(x = relative_access[train_idx, ], file = "data/training_data_regression_1kb_rel.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = relative_access[test_idx, ], file = "data/test_data_regression_1kb_rel.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
