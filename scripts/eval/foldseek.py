import os
import subprocess
import argparse
import csv
import shutil
import pandas as pd


EASY_SEARCH_OUTPUT_COLS = [
    "query",
    "target",
    "theader",
    "evalue",
    "alntmscore",
    "qtnscore",
    "ttmscore",
    "rmsd",
    "lddt",
    "prob",
]


#####################################################################################################
# Diversity scoring
#####################################################################################################


def diversity_scores(
    filename,
):  # Report number of clusters, number of samples. - note representative is included as a cluster member.
    df = pd.read_csv(filename, sep="\t", names=["representative", "member"])
    number_clusters = len(set(df["representative"]))
    number_samples = len(set(df["member"]))
    return (number_clusters, number_samples)


def filter(input_folder, filtered_list, outputfolder):
    print("applying filter")
    tmp_folder = outputfolder + "/temp_foldseek/"
    # make enw temp folder
    os.makedirs(tmp_folder)
    for file in os.listdir(input_folder):
        if input_folder + file in filtered_list:
            # move file to new folde
            shutil.copyfile(input_folder + "/" + file, tmp_folder + "/" + file)
    print("filter applied")

    return tmp_folder


def save_output(data, o):
    out = csv.writer(open(o, "w"), delimiter=",", quoting=csv.QUOTE_ALL)
    if "diversity" in o:
        out.writerow(["input", "nclusters", "nsamples"])
    out.writerow(data)
    print("Analysis outputs saved ", o)


def easy_cluster(i, o, outputfolder, filtered):
    if not os.path.exists(o + "_cluster.tsv"):
        if filtered != None:
            cmd = [
                "foldseek",
                "easy-cluster",
                outputfolder + "temp_foldseek/",
                o,
                "tmp",
                "--alignment-type",
                "2",
                "--tmscore-threshold",
                "0.5",
            ]
        else:
            cmd = [
                "foldseek",
                "easy-cluster",
                i,
                o,
                "tmp",
                "--alignment-type",
                "2",
                "--tmscore-threshold",
                "0.5",
            ]
        subprocess.run(cmd)
    else:
        print("Clustering of samples has already been run ", o + "_cluster.tsv")
    if not os.path.exists(o.split("/")[-1] + "_diversity_score.csv"):
        nclusters, nsamples = diversity_scores(o + "_cluster.tsv")
        # o is the cluster output file path e.g. /homefs/home/robins21/fold_seek_results_/easycluster/test_diversity_score.csv'
        save_output(
            [i, nclusters, nsamples],
            outputfolder + "/scoring/" + o.split("/")[-1] + "_diversity_score.csv",
        )
    else:
        print(
            "Diversity has already been run."
        )


def run_foldseek_easycluster(
    input_folder, outputfile, next_folder, outputfolder, filtered
):
    cluster_output_folder = outputfolder + "/easycluster/"
    if not os.path.exists:
        os.makedirs(cluster_output_folder)
    # make this folder
    if next_folder == None:
        # may need extra novelty filter here
        easy_cluster(
            input_folder, cluster_output_folder + outputfile, outputfolder, filtered
        )
    if next_folder != None:
        for f in os.listdir(input_folder):
            if not os.path.exists(input_folder + f + next_folder):
                continue
            # may need to add extra novelty filter here
            if filtered != None:
                filter(input_folder + f + next_folder, filtered, outputfolder)
            easy_cluster(
                input_folder + "/" + f + "/" + next_folder,
                cluster_output_folder + outputfile + "_" + f,
                outputfolder,
                filtered,
            )
            if os.path.exists(outputfolder + "/temp_foldseek/"):
                shutil.rmtree(outputfolder + "/temp_foldseek/")


#####################################################################################################
# Novelty scoring
#####################################################################################################


def novelty_scores_tm(
    filtered_df,
):  # have df sorted filtered_e = pdb_results.loc[pdb_results['evalue']>=0.5].sort_values(by=['query','evalue'], ascending=False)
    results = []
    filtered_df = filtered_df.loc[filtered_df["alntmscore"] >= 0.5].sort_values(
        by=["query", "alntmscore"], ascending=False
    )
    for i in set(filtered_df["query"]):
        top1 = filtered_df.loc[filtered_df["query"] == i][
            "alntmscore"
        ].max()  # get max evalue
        top5 = filtered_df.loc[filtered_df["query"] == i].head(5)["alntmscore"].mean()
        top10 = (
            filtered_df.loc[filtered_df["query"] == i].head(10)["alntmscore"].mean()
        )  # get max evalue
        results.append(
            [
                i,
                top1,
                top5,
                top10,
                filtered_df.loc[filtered_df["query"] == i].head(10)["theader"].tolist(),
            ]
        )
    x = pd.DataFrame(results, columns=["input", "tm1", "tm5", "tm10", "tm10pdb_hits"])
    return x


def novelty_scores_e(
    filtered_df,
):  # have df sorted filtered_e = pdb_results.loc[pdb_results['evalue']>=0.5].sort_values(by=['query','evalue'], ascending=False)
    filtered_df = filtered_df.loc[filtered_df["evalue"] >= 0.5].sort_values(
        by=["query", "evalue"], ascending=False
    )
    results = []
    for i in set(filtered_df["query"]):
        top1 = filtered_df.loc[filtered_df["query"] == i][
            "evalue"
        ].max()  # get max evalue
        top5 = filtered_df.loc[filtered_df["query"] == i].head(5)["evalue"].mean()
        top10 = (
            filtered_df.loc[filtered_df["query"] == i].head(10)["evalue"].mean()
        )  # get max evalue
        results.append(
            [
                i,
                top1,
                top5,
                top10,
                filtered_df.loc[filtered_df["query"] == i].head(10)["theader"].tolist(),
            ]
        )
    x = pd.DataFrame(results, columns=["input", "e1", "e5", "e", "e10pdb_hits"])
    return x


def easy_search(i, o, filtered):
    if filtered != None:
        i = "/homefs/home/robins21/fold_seek_results_/temp_foldseek/"

    cmd = [
        "foldseek",
        "easy-search",
        i,
        "/data/bucket/robins21/pdb",
        "tmp",
        o,
        "--alignment-type",
        "1",
        "--format-output",
        EASY_SEARCH_OUTPUT_COLS
    ]
    subprocess.run(cmd)


def filterdf(filename):
    pdb_results = pd.read_csv(
        filename,
        sep="\t",
        names=EASY_SEARCH_OUTPUT_COLS
    )
    return pdb_results


def calculate_novelty(pdb_cluster_results_tsv, outputfile, input_folder):
    filtered_df = filterdf(pdb_cluster_results_tsv)
    x1 = novelty_scores_tm(filtered_df)
    x2 = novelty_scores_e(filtered_df)
    # merge the above dfs by same
    x = pd.merge(x1, x2, on="input")
    x.insert(0, "input_folder", [input_folder] * len(x))
    # save to csv
    x.to_csv(outputfile + "_novelty_score.csv")


def run_foldseek(input_folder, outputfile, next_folder, outputfolder, filtered):
    if next_folder == None:
        outputfile2 = (
            outputfolder + "/pdb/" + outputfile + "_pdb.tsv"
        )  # if run in dif folder...
        if not os.path.exists(outputfile2):
            easy_search(input_folder, outputfile2, filtered)
        else:
            print("Clustering to pdb has already been run: ", outputfile2)
        ### Novelty scoring ###
        if not os.path.exists(
            outputfolder + "/scoring/" + outputfile + "_novelty_score.csv"
        ):
            calculate_novelty(
                outputfile2, outputfolder + "/scoring/" + outputfile, input_folder
            )  # calculate novelty - if the above outputfile exists
        else:
            print(
                "Novelty scoring to pdb has already been run: ",
                outputfolder + "/scoring/" + outputfile + "_novelty_score.csv",
            )

    if next_folder != None:
        for f in os.listdir(input_folder):
            # here i make the filtered folder
            outputfile2 = outputfolder + "/pdb/" + outputfile + "_" + f + "_pdb.tsv"

            if not os.path.exists(input_folder + f + next_folder):
                continue  # sometimes subfolders do not have generated structures
            if filtered != None:
                filter(input_folder + f + next_folder, filtered, outputfolder)

            if not os.path.exists(outputfile2):
                easy_search(input_folder + f + next_folder, outputfile2, filtered)
            else:
                print("Clustering to pdb has already been run: ", outputfile2)

            ### Novelty scoring ###
            f_output = outputfolder + "/scoring/" + outputfile + "_" + f
            if not os.path.exists(f_output + "_novelty_score.csv"):
                calculate_novelty(outputfile2, f_output, input_folder + f + next_folder)
            else:
                print(
                    "Novelty scoring to pdb has already been run: ",
                    f_output + "_novelty_score.csv",
                )
            if os.path.exists(outputfolder + "/temp_foldseek/"):
                shutil.rmtree(outputfolder + "/temp_foldseek/")


#####################################################################################################
# Main
#####################################################################################################


def run_foldseek_and_analysis(
    input_folder: str,
    next_folder: str,
    outputfolder: str,
    outputfile: str,
    run_novelty: bool=True,
    run_diversity: bool=True,
    use_filter: bool=True
):
    filtered = None

    # remove temp folder created from 
    if os.path.exists(outputfolder + "/temp_foldseek/"):
        shutil.rmtree(outputfolder + "/temp_foldseek/")
    
    # apply filter, if needed
    if use_filter:
        print("Using developability filter")
        if os.path.exists(input_folder + "/designability.csv"):
            df = pd.read_csv(input_folder + "/designability.csv")

        elif os.path.exists(
            input_folder.replace("generated/structures/", "") + "designability.csv"
        ):
            df = pd.read_csv(
                input_folder.replace("generated/structures/", "") + "designability.csv"
            )
            print("filtered df ", df)
        else:
            print("cant find designability file")
            return

        filtered = df.loc[df["designable"] == True]["pdb_paths"].to_list()

        if next_folder == None:
            # move filtered files into a new folder
            filter(input_folder, filtered, outputfolder)
            print(len(filtered), " proteins remaining")

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # if scoring folder does not exist, create it:
    if not os.path.exists(outputfolder + "/scoring/"):
        os.makedirs(outputfolder + "/scoring/")
    
    if run_novelty:
        print("Running novelty assessment ")
        run_foldseek(input_folder, outputfile, next_folder, outputfolder, filtered)

    if run_diversity:
        print("Running diversity assessment ")
        run_foldseek_easycluster(
            input_folder, outputfile, next_folder, outputfolder, filtered
        )
    # if os.path.exists(outputfolder+'/temp_foldseek/'):shutil.rmtree(outputfolder+'/temp_foldseek/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runfoldseekcluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-input_folder", type=str)
    parser.add_argument("--next_folder", type=str)
    parser.add_argument("-outputfolder", type=str)
    parser.add_argument("-outputfile", type=str)
    parser.add_argument("--n", action="store_true")  # novelty_cluster_flag
    parser.add_argument("--d", action="store_true")  # diversity_cluster_flag
    parser.add_argument("--f", action="store_true")
    # pass filter?
    args = parser.parse_args()
    print(args)
    run_foldseek_and_analysis(**vars(args))
