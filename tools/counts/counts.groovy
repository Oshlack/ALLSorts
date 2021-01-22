/*==================================================================================================

	ALLSorts Counts Generator
	
	Generate the optimal input for ALL. Start from fastq, fastq.gz, or bam!
	
	Author: Breon Schmidt
	Version: 0.1
	Date: 16 DEC 2020

==================================================================================================*/

core = System.getenv("COUNTSDIR")
STAR = System.getenv("star")
//SAMTOOLS = System.getenv("samtools")
FEATURECOUNTS = System.getenv("featurecounts")

/*===========================================================
	Get user defined variables or define defaults
===========================================================*/

if(!binding.variables.containsKey("results")){
	print("Please enter a destination for the ouput. i.e. -p results='destination/to/results/'")
	System.exit(0)
}

if(!binding.variables.containsKey("strand")){
	print("Please indicate whether the alignments are unstranded/reverse/forward strand.'")
	print("i.e. -p stranded=no, (no, reverse, forward)")
	System.exit(0)
}

if(!binding.variables.containsKey("a_mem")){
    if((type == "fasta") || (type=="fastq")){
        print("Please enter the amount of memory (bytes) for the alignment step.'")
        print("i.e. -p a_mem=64000000000")
        System.exit(0)
    }
}

if(!binding.variables.containsKey("classify")){
	classify="no"
} 

if(!binding.variables.containsKey("threads")){
	threads=1
}

if(!binding.variables.containsKey("genome_fa")){
	genome_fasta=core+"/resources/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
}

if(!binding.variables.containsKey("genome_dir")){
    genome_dir=core+"/genome/"
}

if(!binding.variables.containsKey("reference")){
    reference=core+"/resources/Homo_sapiens.GRCh37.87.chr.gtf"
}

if(!binding.variables.containsKey("type")){
	print("Please enter a value for the input format. i.e. -p type=fastq/fasta/bam'")
	System.exit(0)
}

if(!binding.variables.containsKey("format")){
	if(type == "fastq"){
		format="*/%_*.fastq.gz"
	} else if(type == "bam") {
		format="*/%_Aligned.sortedByCoord.out.bam"
	}
}

/*===========================================================
	Setup directories based on output folder
===========================================================*/

alignments = results+"/alignments/"
counts = results+"/counts/"
predictions = results+"/predictions/"

/*===========================================================
	Stages
===========================================================*/

// ### Setup

make_dir = {
	produce("$results/info.log"){
		exec """mkdir $alignments;
				mkdir $counts;
				mkdir $predictions;
				touch $results/info.log""", single
	}
}

star_align = {

	output.dir = alignments

	from("*.fastq.gz") produce(branch.name+"_Aligned.sortedByCoord.out.bam"){
		def sample_name = branch.name+"_"
		def prefix = alignments+sample_name

		exec """$STAR --genomeDir $genome_dir
					  --runMode alignReads
					  --twopassMode Basic 
					  --readFilesIn $input1 $input2
					  --readFilesCommand gunzip -c
					  --quantMode GeneCounts
					  --runThreadN $threads
					  --limitBAMsortRAM $a_mem 
					  --outFileNamePrefix $prefix
					  --chimOutType WithinBAM
					  --outSAMtype BAM SortedByCoordinate""", align
	}
}

star_align_fasta = {

	output.dir = alignments

	from("*.fasta.gz") produce(branch.name+"_Aligned.sortedByCoord.out.bam"){
		def sample_name = branch.name+"_"
		def prefix = alignments+sample_name

		exec """$STAR --genomeDir $genome_dir
					  --runMode alignReads
					  --twopassMode Basic
					  --readFilesIn $input1 $input2
					  --readFilesCommand gunzip -c
					  --quantMode GeneCounts
					  --runThreadN $threads
					  --limitBAMsortRAM $a_mem
					  --outFileNamePrefix $prefix
					  --chimOutType WithinBAM
					  --outSAMtype BAM SortedByCoordinate""", align
	}
}

sam_index = {

	output.dir = alignments
	from(".bam") produce(branch.name+"_Aligned.sortedByCoord.out.bam.bai"){		
		exec """samtools index $input.bam""", index
	
	}

}

star_count = {

	output.dir = counts
	produce ("counts.csv") {
		exec """python $core/scripts/counts_star.py 
						-strand $strand 
						-directory $alignments
						-output $counts/counts.csv
			""", single
	}
}

feature_counts = {
	output.dir = counts
	from(".bam") produce("feature_counts.csv"){
		exec """$FEATURECOUNTS -p -t exon -F GTF -a $reference -T $threads -o $counts/feature_counts.csv $input""", featurecounts
	}
}

fc_count = {
	output.dir = counts
	produce ("counts.csv") {
		exec """python $core/scripts/counts_fc.py 
						-counts $counts/feature_counts.csv
						-output $counts/counts.csv
			""", single
	}
}

classify = {
    output.dir = predictions
	from ("counts.csv") produce ("predictions.csv") {
		exec """ALLSorts -s $input -d $predictions -ball True -parents""", single
	}
}


/*===========================================================
	Stages
===========================================================*/

run { 

    if (type == "fastq") {
	    make_dir + format*[star_align + sam_index] + star_count + classify
	} else if (type == "bam") {
        make_dir + feature_counts + fc_count + classify
	} else if (type == "fasta") {
	    make_dir + format*[star_align_fasta + sam_index] + star_count + classify
	}

}
