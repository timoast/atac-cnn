library(Signac)
library(Seurat)

pbmc <- readRDS("data/pbmc_multiomic.rds")

# binarize counts, average for each group (fraction of cells with >0 counts)
count.matrix <- GetAssayData(object = pbmc, assay = "ATAC", slot = "counts")
binary.counts <- BinarizeCounts(object = count.matrix)
# create new assay
pbmc[['binary']] <- CreateChromatinAssay(counts = binary.counts)
avg.raw <- Signac:::AverageCountMatrix(object = pbmc, assay = "binary")

# # look at distribution of fractions to decide on cutoff for open/closed
# hist(avg.cells, 100)
# 
# frag <- Fragments(pbmc)
# frag <- UpdatePath(object = frag[[1]], new.path = "/home/stuartt/atac-cnn/data/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz")
# 
# Fragments(pbmc) <- NULL
# Fragments(pbmc) <- frag
# 
# CoverageBrowser(pbmc, region = "chr1-181323-181605")

# set peaks open in >10% of cells as "open" (1) otherwise "closed" (0)
avg.cells <- ifelse(avg.cells > 0.1, 1, 0)

# randomize order
avg.cells <- avg.cells[sample(rownames(avg.cells), size = nrow(avg.cells), replace = FALSE), ]

# split into train, test
n_examples <- nrow(avg.cells)
train_idx <- rownames(avg.cells)[1:(0.8*n_examples)]
test_idx <- setdiff(rownames(avg.cells), train_idx)
write.table(x = avg.cells[train_idx, ], file = "data/training_data_v1.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
write.table(x = avg.cells[test_idx, ], file = "data/test_data_v1.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
