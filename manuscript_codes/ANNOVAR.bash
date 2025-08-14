#!/bin/bash
#SBATCH --job-name=ANNOVAR_batch
#SBATCH --mail-user=Po-Yu.Lin@nyulangone.org
#SBATCH --mail-type=ALL
#SBATCH --output=/gpfs/home/pl2948/VariantInterpretation/log/ANNOVAR_%j.out
#SBATCH --error=/gpfs/home/pl2948/VariantInterpretation/log/ANNOVAR_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu_de#SBATCH --job-name=ANNOVAR_batch
#SBATCH --mail-user=Po-Yu.Lin@nyulangone.org
#SBATCH --mail-type=ALL
#SBATCH --output=/gpfs/home/pl2948/VariantInterpretation/log/ANNOVAR_%j.out
#SBATCH --error=/gpfs/home/pl2948/VariantInterpretation/log/ANNOVAR_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu_dev

cd /gpfs/home/pl2948/VariantInterpretation/Data

INPUTS=("ClinVar2019.avinput" "ClinVar2020.avinput" "GnomADSet.avinput")
DB_DIR="/gpfs/home/pl2948/VariantInterpretation/annovar/humandb"
ANNOVAR_DIR="/gpfs/home/pl2948/VariantInterpretation/annovar"

for INPUT in "${INPUTS[@]}"; do
    BASENAME=$(basename "$INPUT" .avinput)

    perl $ANNOVAR_DIR/table_annovar.pl \
        "$INPUT" "$DB_DIR" -buildver hg19 -out "${BASENAME}_dbnsfp52a" \
        -protocol dbnsfp52a \
        -operation f \
        -nastring . -csvout -polish \
        -remove

done

echo "All annotations completed."


