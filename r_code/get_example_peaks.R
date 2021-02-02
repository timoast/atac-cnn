library(Signac)
library(Seurat)
library(BSgenome.Hsapiens.UCSC.hg38)
library(GenomicRanges)
library(ggplot2)

pbmc <- readRDS("data/pbmc_multiomic.rds")

# find differential peaks between mono and T
da.peaks <- FindMarkers(
  object = pbmc,
  ident.1 = "CD14 Mono",
  ident.2 = "CD8 Naive",
  test.use = "LR",
  latent.vars = "nCount_ATAC"
)

head(da.peaks)

CoveragePlot(pbmc, head(rownames(da.peaks)))

# Find DE motifs
motifs <- FindMotifs(
  object = pbmc,
  features = head(rownames(da.peaks), 5000)
)

pbmc <- RunChromVAR(pbmc, genome = BSgenome.Hsapiens.UCSC.hg38)

FeaturePlot(pbmc, paste0('chromvar_', head(rownames(motifs)))) &
  ggplot2::scale_color_viridis_c()

VlnPlot(pbmc, "chromvar_MA0466.2")

FeaturePlot(pbmc, "chromvar_MA0466.2")

# look at DA peaks containing the CEBPB motif
da.up <- da.peaks[da.peaks$avg_log2FC > 0, ]
cebpb.positions <- Motifs(pbmc)@positions$MA0466.2
da.ranges <- StringToGRanges(rownames(head(da.up, 5000)))
da.with.motif <- subsetByOverlaps(da.ranges, cebpb.positions)
da.use <- da.peaks[GRangesToString(da.with.motif), ]

CoveragePlot(
  object = pbmc,
  region = head(rownames(da.use), 3),
  ranges = resize(cebpb.positions, width = 50, fix = "center"),
  ncol = 3,
  extend.upstream = 500,
  extend.downstream = 500
) + ggsave("./covplot_cebpb.png", height = 6, width = 12)


MotifPlot(pbmc, motifs = "MA0466.2") +
  ggsave("./cebpb.png")

# write DA peaks to a separate file in same format as training data
# so we can easily iterate over them
val.data <- read.table("data/test_data_regression_1kb_rel.tsv", sep="\t")
olap.test <- subsetByOverlaps(StringToGRanges(rownames(val.data)), da.with.motif)
peaks.motif <- val.data[GRangesToString(olap.test), ]

# find the coordinates of the motif within each peak
positions <- findOverlaps(query = olap.test, subject = cebpb.positions)

head(cebpb.positions[subjectHits(positions)])

find_relative_position <- function(motifs, peaks) {
  # output a dataframe of peak ID, motif position within peak
  df <- data.frame()
  # overlap
  olaps <- findOverlaps(query = peaks, subject = motifs)
  # for each overlapping peak, get the overlapping motif coordinates
  unique_hits <- unique(queryHits(olaps))
  for (i in unique_hits) {
    # get overlapping 
    motif_hits <- subjectHits(olaps)[queryHits(olaps) == i]
    motif_positions <- motifs[motif_hits]
    relative_position <- start(motif_positions) - start(peaks)[i]
    if (length(relative_position) == 1) {
      # only retain cases where there's a single motif in the peak
      for (j in unique(relative_position)) {
        df <- rbind(df, data.frame("peak" = GRangesToString(peaks[i]), "motif_start" = j))
      }
    }
  }
  return(df)
}

df <- find_relative_position(motifs = cebpb.positions, peaks = olap.test)
write.table(df, file = "data/motif_positions_within_peak.tsv", sep="\t", col.names = TRUE, row.names = FALSE)

# ensure the order is exact same as peak file
a <- peaks.motif[df$peak, ]
write.table(x = a, file = "data/val_da_peaks_with_cebpb.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")

# Do the same, but for a different cell type

da.peaks <- FindMarkers(
  object = pbmc,
  ident.1 = c("Naive B", "Intermediate B", "Memory B"),
  # test.use = "LR",
  # latent.vars = "nCount_ATAC"
)

up_b <- da.peaks[da.peaks$avg_log2FC > 0, ]

VlnPlot(pbmc, "chromvar_MA0154.4")

# Find DA peaks containing the EBF1 motif
ebf1.positions <- Motifs(pbmc)@positions$MA0154.4
da.ranges <- StringToGRanges(rownames(head(up_b, 5000)))
da.with.motif <- subsetByOverlaps(da.ranges, ebf1.positions)
da.use <- da.peaks[GRangesToString(da.with.motif), ]

CoveragePlot(
  object = pbmc,
  region = head(rownames(da.use), 3),
  ranges = resize(ebf1.positions, width = 50, fix = "center"),
  ncol = 3,
  extend.upstream = 500,
  extend.downstream = 500
) + ggsave("./covplot_ebf1.png", height = 6, width = 12)


MotifPlot(pbmc, motifs = "MA0154.4") +
  ggsave("./ebf1.png")


olap.ebf1 <- subsetByOverlaps(StringToGRanges(rownames(val.data)), da.with.motif)

df.ebf1 <- find_relative_position(motifs = ebf1.positions, peaks = olap.ebf1)
write.table(df.ebf1, file = "data/ebf1_positions_within_peak.tsv", sep="\t", col.names = TRUE, row.names = FALSE)

# ensure the order is exact same as peak file
peaks.motif <- val.data[GRangesToString(olap.ebf1), ]
a <- peaks.motif[df.ebf1$peak, ]
write.table(x = a, file = "data/val_da_peaks_with_ebf1.tsv", col.names = TRUE, row.names = TRUE, quote = FALSE, sep = "\t")
