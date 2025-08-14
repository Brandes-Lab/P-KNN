#!/bin/bash
#SBATCH --job-name=ANNOVAR_gnomAD
#SBATCH --mail-user=Po-Yu.Lin@nyulangone.org
#SBATCH --mail-type=ALL
#SBATCH --output=/gpfs/home/pl2948/ASI/log/ANNOVAR_%j.out
#SBATCH --error=/gpfs/home/pl2948/ASI/log/ANNOVAR_%j.err
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G 
#SBATCH --time=4:00:00  
#SBATCH --partition=cpu_dev

perl /gpfs/home/pl2948/VariantInterpretation/annovar/table_annovar.pl \
    /gpfs/home/pl2948/VariantInterpretation/Data/ClinVar2025.avinput \
    /gpfs/home/pl2948/VariantInterpretation/annovar/humandb/ \
    -buildver hg38 \
    -out /gpfs/home/pl2948/VariantInterpretation/Data/ClinVar2025_gnomAD \
    -protocol gnomad211_exome,gnomad211_genome \
    -operation f,f \
    -nastring . -csvout -polish -remove

echo "ANNOVAR annotation completed successfully."
