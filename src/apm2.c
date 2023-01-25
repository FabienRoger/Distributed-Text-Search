/**
 * APPROXIMATE PATTERN MATCHING
 *
 * INF560
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#define APM_DEBUG 0

char *
read_input_file(char *filename, int *size)
{
    char *buf;
    off_t fsize;
    int fd = 0;
    int n_bytes = 1;

    /* Open the text file */
    fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        fprintf(stderr, "Unable to open the text file <%s>\n", filename);
        return NULL;
    }

    /* Get the number of characters in the textfile */
    fsize = lseek(fd, 0, SEEK_END);
    if (fsize == -1)
    {
        fprintf(stderr, "Unable to lseek to the end\n");
        return NULL;
    }

#if APM_DEBUG
    printf("File length: %lld\n", fsize);
#endif

    /* Go back to the beginning of the input file */
    if (lseek(fd, 0, SEEK_SET) == -1)
    {
        fprintf(stderr, "Unable to lseek to start\n");
        return NULL;
    }

    /* Allocate data to copy the target text */
    buf = (char *)malloc(fsize * sizeof(char));
    if (buf == NULL)
    {
        fprintf(stderr, "Unable to allocate %lld byte(s) for main array\n",
                fsize);
        return NULL;
    }

    n_bytes = read(fd, buf, fsize);
    if (n_bytes != fsize)
    {
        fprintf(stderr,
                "Unable to copy %lld byte(s) from text file (%d byte(s) copied)\n",
                fsize, n_bytes);
        return NULL;
    }

#if APM_DEBUG
    printf("Number of read bytes: %d\n", n_bytes);
#endif

    *size = n_bytes;

    close(fd);

    return buf;
}

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
#define MIN2(a, b) (a < b ? a : b)

int levenshtein(char *s1, char *s2, int len, int *column)
{
    unsigned int x, y, lastdiag, olddiag;

    for (y = 1; y <= len; y++)
    {
        column[y] = y;
    }
    for (x = 1; x <= len; x++)
    {
        column[0] = x;
        lastdiag = x - 1;
        for (y = 1; y <= len; y++)
        {
            olddiag = column[y];
            column[y] = MIN3(
                column[y] + 1,
                column[y - 1] + 1,
                lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return (column[len]);
}

int main(int argc, char **argv)
{
    char **pattern;
    char *filename;
    int approx_factor = 0;
    int nb_patterns = 0;
    int i, j;
    char *buf;
    char *own_buf;
    struct timeval t1, t2;
    double duration;
    int n_bytes;
    int own_n_bytes;
    int *n_matches;
    int rank, comm_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    /* Check number of arguments */
    if (argc < 4)
    {
        printf("Usage: %s approximation_factor "
               "dna_database pattern1 pattern2 ...\n",
               argv[0]);
        return 1;
    }

    /* Get the distance factor */
    approx_factor = atoi(argv[1]);

    /* Grab the filename containing the target text */
    filename = argv[2];

    /* Get the number of patterns that the user wants to search for */
    nb_patterns = argc - 3;

    /* Fill the pattern array */
    pattern = (char **)malloc(nb_patterns * sizeof(char *));
    if (pattern == NULL)
    {
        fprintf(stderr,
                "Unable to allocate array of pattern of size %d\n",
                nb_patterns);
        return 1;
    }

    /* Grab the patterns */
    for (i = 0; i < nb_patterns; i++)
    {
        int l;

        l = strlen(argv[i + 3]);
        if (l <= 0)
        {
            fprintf(stderr, "Error while parsing argument %d\n", i + 3);
            return 1;
        }

        pattern[i] = (char *)malloc((l + 1) * sizeof(char));
        if (pattern[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate string of size %d\n", l);
            return 1;
        }

        strncpy(pattern[i], argv[i + 3], (l + 1));
    }

    if (rank == 0)
        printf("Approximate Pattern Mathing: "
               "looking for %d pattern(s) in file %s w/ distance of %d\n",
               nb_patterns, filename, approx_factor);

    buf = read_input_file(filename, &n_bytes);
    if (buf == NULL)
    {
        return 1;
    }

    /* Allocate the array of matches */
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    if (n_matches == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                nb_patterns * sizeof(int));
        return 1;
    }

    /*****
     * BEGIN MAIN LOOP
     ******/

    /* Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);

    // gather all files in the same buffer
    // first get the lengths of each file
    int *lengths = (int *)malloc(comm_size * sizeof(int));
    MPI_Allgather(&own_n_bytes, 1, MPI_INT, lengths, 1, MPI_INT, MPI_COMM_WORLD);

    // Allocate memory for the displacement array
    int *displs = (int *)malloc(comm_size * sizeof(int));
    // Fill the displacement array with the offsets for each process' data in the concatenated table
    n_bytes = 0;
    int own_start = 0;
    int own_end = 0;
    for (i = 0; i < comm_size; i++)
    {
        n_bytes += lengths[i];
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + lengths[i - 1]);
        if (i == rank)
        {
            own_start = displs[i];
            own_end = displs[i] + lengths[i];
        }
    }

    // finds longest pattern
    int max_pattern_length = 0;
    for (i = 0; i < nb_patterns; i++)
    {
        if (strlen(pattern[i]) > max_pattern_length)
        {
            max_pattern_length = strlen(pattern[i]);
        }
    }

    // send data to people who need it asynchronically
    for (i = 0; i < comm_size; i++)
    {
        int other_start = displs[i] / sizeof(int);
        int other_end_data = MIN2((rank + 1) * n_bytes / comm_size + max_pattern_length, n_bytes);

        if (other_start < own_end && other_end_data > own_start)
        {
            int start = MAX2(other_start, own_start);
            int end = MIN2(other_end_data, own_end);
            int length = end - start;
            MPI_Isend(buf + start, length, MPI_CHAR, i, 0, MPI_COMM_WORLD, NULL);
        }
    }

    int start = rank * n_bytes / comm_size;
    int end_data = MIN2((rank + 1) * n_bytes / comm_size, n_bytes) + max_pattern_length;
    int end = MIN2((rank + 1) * n_bytes / comm_size, n_bytes);

    // Allocate memory for the concatenated table on process 0
    if (rank == 0)
    {
        buf = (int *)malloc(n_bytes * sizeof(int));
    }

    // then gather all the files in the same buffer
    MPI_Gatherv(own_buf, own_n_bytes, MPI_CHAR, buf, lengths, displs, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        MPI_Finalize();
        return 0;
    }

    /* Check each pattern one by one */
    for (i = 0; i < nb_patterns; i++)
    {
        int size_pattern = strlen(pattern[i]);
        int *column;

        /* Initialize the number of matches to 0 */
        n_matches[i] = 0;

        column = (int *)malloc((size_pattern + 1) * sizeof(int));
        if (column == NULL)
        {
            fprintf(stderr, "Error: unable to allocate memory for column (%ldB)\n",
                    (size_pattern + 1) * sizeof(int));
            return 1;
        }

        /* Traverse the input data up to the end of the file */
        for (j = start; j < end; j++)
        {
            int distance = 0;
            int size;

#if APM_DEBUG
            if (j % 100 == 0)
            {
                printf("Procesing byte %d (out of %d)\n", j, n_bytes);
            }
#endif

            size = size_pattern;
            if (n_bytes - j < size_pattern)
            {
                size = n_bytes - j;
            }

            distance = levenshtein(pattern[i], &buf[j], size, column);

            if (distance <= approx_factor)
            {
                n_matches[i]++;
                // printf("Match found at position %d for pattern <%s> (distance: %d)\n",
                //        j, pattern[i], distance);
            }
        }

        free(column);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // reduce the number of matches
    for (i = 0; i < nb_patterns; i++)
    {
        int tmp;
        MPI_Reduce(&n_matches[i], &tmp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        n_matches[i] = tmp;
    }

    /* Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    if (rank == 0)
        printf("%s done in %lf s (size ; %d)\n", argv[0], duration, comm_size);

    /*****
     * END MAIN LOOP
     ******/
    if (rank == 0)
        for (i = 0; i < nb_patterns; i++)
        {
            printf("Number of matches for pattern <%s>: %d\n",
                   pattern[i], n_matches[i]);
        }
    MPI_Finalize();

    return 0;
}
