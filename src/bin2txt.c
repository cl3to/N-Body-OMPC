#include <stdio.h>
#include "include/body.h"

#define NFILES 4

int main(int argc, char **argv)
{
	const char *binFilenames[NFILES] = {
		"../debug/computed_pos_12",
		"../debug/solution_pos_12",
		"../debug/computed_vel_12",
		"../debug/solution_vel_12"};
	const char *txtFilenames[NFILES] = {
		"../debug/computed_pos_12.txt",
		"../debug/solution_pos_12.txt",
		"../debug/computed_vel_12.txt",
		"../debug/solution_vel_12.txt"};

	for (int i = 0; i < NFILES; i++)
	{
		FILE *binFile = fopen(binFilenames[i], "rb");
		FILE *txtFile = fopen(txtFilenames[i], "w");

		if (binFile == NULL || txtFile == NULL)
		{
			perror("Error opening file");
			return 1;
		}

		Pos temp;
		while (fread(&temp, sizeof(Pos), 1, binFile))
		{ // Read struct from binary
			fprintf(txtFile, "%.9g %.9g %.9g\n", temp.x, temp.y, temp.z);
		}

		fclose(binFile);
		fclose(txtFile);
	}

	return 0;
}
