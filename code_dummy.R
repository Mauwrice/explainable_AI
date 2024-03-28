library(readr)
library(caret)
# Read the CSV file
baseline_data_zurich_prepared <- read_csv("baseline_data_zurich_prepared.csv")

# Convert specific variables to factors
baseline_data_zurich_prepared$mrs_before <- factor(baseline_data_zurich_prepared$mrs_before, 
                                                   levels = sort(unique(baseline_data_zurich_prepared$mrs_before)),
                                                   labels = c(0,1,2,3,4))

baseline_data_zurich_prepared$mrs3 <- factor(baseline_data_zurich_prepared$mrs3, 
                                             levels = sort(unique(baseline_data_zurich_prepared$mrs3)),
                                             labels = c(0,1,2,3,4,5,6))

baseline_data_zurich_prepared$stroke_beforey <- factor(baseline_data_zurich_prepared$stroke_beforey,
                                                       levels = sort(unique(baseline_data_zurich_prepared$stroke_beforey)),
                                                       labels = c(0,1))

baseline_data_zurich_prepared$tia_beforey <- factor(baseline_data_zurich_prepared$tia_beforey,
                                                    levels = sort(unique(baseline_data_zurich_prepared$tia_beforey)),
                                                    labels = c(0,1))

baseline_data_zurich_prepared$rf_atrial_fibrillationy <- factor(baseline_data_zurich_prepared$rf_atrial_fibrillationy,
                                                                levels = sort(unique(baseline_data_zurich_prepared$rf_atrial_fibrillationy)),
                                                                labels = c(0,1))

baseline_data_zurich_prepared$ich_beforey <- factor(baseline_data_zurich_prepared$ich_beforey,
                                                    levels = sort(unique(baseline_data_zurich_prepared$ich_beforey)),
                                                    labels = c(0,1))

baseline_data_zurich_prepared$rf_hypertoniay <- factor(baseline_data_zurich_prepared$rf_hypertoniay,
                                                       levels = sort(unique(baseline_data_zurich_prepared$rf_hypertoniay)),
                                                       labels = c(0,1))

baseline_data_zurich_prepared$rf_diabetesy <- factor(baseline_data_zurich_prepared$rf_diabetesy,
                                                     levels = sort(unique(baseline_data_zurich_prepared$rf_diabetesy)),
                                                     labels = c(0,1))

baseline_data_zurich_prepared$rf_hypercholesterolemiay <- factor(baseline_data_zurich_prepared$rf_hypercholesterolemiay,
                                                                 levels = sort(unique(baseline_data_zurich_prepared$rf_hypercholesterolemiay)),
                                                                 labels = c(0,1))

baseline_data_zurich_prepared$rf_smokery <- factor(baseline_data_zurich_prepared$rf_smokery,
                                                   levels = sort(unique(baseline_data_zurich_prepared$rf_smokery)),
                                                   labels = c(0,1))

baseline_data_zurich_prepared$rf_chdy <- factor(baseline_data_zurich_prepared$rf_chdy,
                                                levels = sort(unique(baseline_data_zurich_prepared$rf_chdy)),
                                                labels = c(0,1))

baseline_data_zurich_prepared$eventtia <- factor(baseline_data_zurich_prepared$eventtia,
                                                 levels = sort(unique(baseline_data_zurich_prepared$eventtia)),
                                                 labels = c(0,1))

baseline_data_zurich_prepared$sexm <- factor(baseline_data_zurich_prepared$sexm,
                                             levels = sort(unique(baseline_data_zurich_prepared$sexm)),
                                             labels = c(0,1))


dmy <- dummyVars(" ~ .", data = baseline_data_zurich_prepared)
trsf <- data.frame(predict(dmy, newdata = baseline_data_zurich_prepared))
trsf

write_csv(trsf, "data/baseline_data_transformed.csv")
