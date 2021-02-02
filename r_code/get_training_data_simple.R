library(GenomicRanges)
library(Signac)
library(Seurat)
library(Matrix)

pbmc <- readRDS("data/pbmc_multiomic.rds")

# remove low count cell types
cell_counts <- table(Idents(pbmc))

# pbmc <- pbmc[, pbmc$celltype %in% names(cell_counts[cell_counts > 50])]

normalize_counts <- function(object, assay = "ATAC", group.by = "celltype") {
  counts <- GetAssayData(object = object, assay = assay, slot = "counts")
  
  # get widths
  widths <- width(StringToGRanges(rownames(counts)))
  
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
  
  # divide by width
  collapsed.counts.norm <- t(crossprod(collapsed.counts, Diagonal(x = 1/widths)))
  
  # divide by scale factors to account for cell number and depth
  collapsed.counts.scaled <- tcrossprod(collapsed.counts.norm, Diagonal(x = 1/scale.factors[colnames(collapsed.counts.norm)]))
  
  # multiply by constant
  collapsed.counts.scaled <- collapsed.counts.scaled * 1e8
  return(as.matrix(collapsed.counts.scaled))
}

avg.counts <- normalize_counts(object = pbmc)

CoveragePlot(pbmc, region = rownames(avg.counts)[200], extend.upstream = 5000, extend.downstream = 5000)

classifications <- ifelse(avg.counts > 0.5, 1, 0)

CoveragePlot(pbmc, region = rownames(classifications)[5], extend.upstream = 5000, extend.downstream = 5000)
# 
# # resize peaks to the same width
# resized <- resize(x = StringToGRanges(rownames(classifications)), width = 1000, fix = 'center')
# 
# # rename rows in matrix
# rownames(classifications) <- GRangesToString(resized)
# 
# # remove cell types with low number of cells
# classifications <- classifications[, names(cell_counts[cell_counts > 50])]
# 
# total_class <- rowSums(classifications)
# classifications <- classifications[(total_class > 0) & (total_class < ncol(classifications)), ]

# write matrix to file

# randomize order
classifications <- classifications[sample(rownames(classifications), size = nrow(classifications), replace = FALSE), ]

# split into train, test
n_examples <- nrow(classifications)
train_idx <- rownames(classifications)[1:(0.8*n_examples)]
test_idx <- setdiff(rownames(classifications), train_idx)
write.table(x = classifications[train_idx, ], file = "data/training_data_v4.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = classifications[test_idx, ], file = "data/test_data_v4.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")


# write continuous version for regression
# split into train, test
n_examples <- nrow(avg.counts)
avg.counts <- avg.counts[sample(rownames(avg.counts), size = nrow(avg.counts), replace = FALSE), ]
train_idx <- rownames(avg.counts)[1:(0.8*n_examples)]
test_idx <- setdiff(rownames(avg.counts), train_idx)
write.table(x = avg.counts[train_idx, ], file = "data/training_data_regression.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = avg.counts[test_idx, ], file = "data/test_data_regression.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
