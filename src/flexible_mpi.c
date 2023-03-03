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
#include <omp.h>

#define APM_DEBUG 0

int DISTRIBUTE_PATTERNS, ONLY_RANK_0;

int sum_all(int *a, int n);

int get_env_int(char *var_name, int def)
{
    char *env_var = getenv(var_name);
    if (env_var == NULL)
    {
        return def;
    }
    return atoi(env_var);
}

char *read_input_file(char *filename, int *size)
{
    char *buf;
    off_t fsize;
    int fd = 0;
    int n_bytes = 1;

    /* Open the text file */
    fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        // fprintf(stderr, "Unable to open the text file <%s>\n", filename);
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
#define MAX2(a, b) (a > b ? a : b)

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

void fill_data_bounds(int rank, int comm_size, int nb_patterns, int max_pattern_length, int n_bytes, int *start, int *end, int *end_data, int *starti, int *endi)
{
    int starti_, endi_, start_, end_;
    if (ONLY_RANK_0)
    {
        if (rank == 0)
        {
            starti_ = 0;
            endi_ = nb_patterns;
            start_ = 0;
            end_ = n_bytes;
        }
        else
        {
            starti_ = 0;
            endi_ = 0;
            start_ = 0;
            end_ = 0;
        }
    }
    else if (!DISTRIBUTE_PATTERNS)
    {
        starti_ = 0;
        endi_ = nb_patterns;
        start_ = rank * n_bytes / comm_size;
        end_ = MIN2((rank + 1) * n_bytes / comm_size, n_bytes);
    }
    else
    {
        int process_in_my_patterns, rank_in_my_pattern;
        if (comm_size < nb_patterns)
        {
            starti_ = rank * nb_patterns / comm_size;
            endi_ = (rank + 1) * nb_patterns / comm_size;
            process_in_my_patterns = 1;
            rank_in_my_pattern = 0;
        }
        else
        {
            int process_per_pattern = comm_size / nb_patterns;
            starti_ = rank / process_per_pattern;
            endi_ = starti_ + 1;
            process_in_my_patterns = process_per_pattern;
            rank_in_my_pattern = rank % process_per_pattern;
            if (starti_ >= nb_patterns)
            {
                // put in process 0
                starti_ = 0;
                endi_ = 1;
                rank_in_my_pattern += process_per_pattern;
            }
            if (starti_ == 0)
            {
                // leftovers
                process_in_my_patterns += comm_size - nb_patterns * process_per_pattern;
            }
        }

        start_ = rank_in_my_pattern * n_bytes / process_in_my_patterns;
        end_ = MIN2((rank_in_my_pattern + 1) * n_bytes / process_in_my_patterns, n_bytes);
    }

    if (start != NULL)
        *start = start_;
    if (end != NULL)
        *end = end_;
    if (end_data != NULL)
        *end_data = MIN2(end_ + max_pattern_length, n_bytes);
    if (starti != NULL)
        *starti = starti_;
    if (endi != NULL)
        *endi = endi_;
}

int intersect(int start, int end, int start2, int end2)
{
    return (start < end2 && end > start2);
}

void parse_args(int argc, char **argv, int rank, int *approx_factor, char **filename, char ***pattern, int *nb_patterns)
{
    /* Check number of arguments */
    if (argc < 4)
    {
        printf("Usage: %s approximation_factor "
               "dna_database pattern1 pattern2 ...\n",
               argv[0]);
        exit(1);
    }

    /* Get the distance factor */
    *approx_factor = atoi(argv[1]);

    /* Grab the filename containing the target text */
    int len = strlen(argv[2]);
    *filename = (char *)malloc((len + 10) * sizeof(char));
    if (*filename == NULL)
    {
        fprintf(stderr, "Unable to allocate string of size %d\n", len);
        exit(1);
    }
    strncpy(*filename, argv[2], len);
    // concat "/rank.txt" to the filemame (where rank is the rank of the process)
    sprintf(*filename + len, "/%d.txt", rank);

    /* Get the number of patterns that the user wants to search for */
    *nb_patterns = argc - 3;

    /* Fill the pattern array */
    *pattern = (char **)malloc(*nb_patterns * sizeof(char *));
    if (*pattern == NULL)
    {
        fprintf(stderr,
                "Unable to allocate array of pattern of size %d\n",
                *nb_patterns);
        exit(1);
    }

    /* Grab the patterns */
    int i;
    for (i = 0; i < *nb_patterns; i++)
    {
        int l;

        l = strlen(argv[i + 3]);
        if (l <= 0)
        {
            fprintf(stderr, "Error while parsing argument %d\n", i + 3);
            exit(1);
        }

        (*pattern)[i] = (char *)malloc((l + 1) * sizeof(char));
        if ((*pattern)[i] == NULL)
        {
            fprintf(stderr, "Unable to allocate string of size %d\n", l);
            exit(1);
        }

        strncpy((*pattern)[i], argv[i + 3], (l + 1));
    }
}

int longest_string_len(char **strings, int nb_strings)
{
    int i;
    int max = 0;
    for (i = 0; i < nb_strings; i++)
    {
        int len = strlen(strings[i]);
        if (len > max)
            max = len;
    }
    return max;
}

int main(int argc, char **argv)
{
    char **pattern;
    char *filename;
    int approx_factor = 0;
    int nb_patterns = 0;
    int i, j;
    char *buf;
    struct timeval t1, t2;
    double duration;
    int n_bytes;
    int own_n_bytes;
    char *own_buf;
    int *n_matches;
    int rank, comm_size;

    DISTRIBUTE_PATTERNS = get_env_int("DISTRIBUTE_PATTERNS", 1);
    ONLY_RANK_0 = get_env_int("ONLY_RANK_0", 0);

#if APM_DEBUG
    printf("DISTRIBUTE_PATTERNS = %d, argc = %d\n", DISTRIBUTE_PATTERNS, argc);
#endif

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // int *a = (int *)malloc(100 * sizeof(int));
    // for (i = 0; i < 100; i++)
    // {
    //     a[i] = i;
    // }
    // int r = sum_all(a, 100);
    // printf("rank = %d, sum = %d", rank, r);
    // MPI_Finalize();
    // return 0;

#if APM_DEBUG
    if (rank == 0)
        printf("comm_size = %d", comm_size);
#endif

    parse_args(argc, argv, rank, &approx_factor, &filename, &pattern, &nb_patterns);

    if (rank == 0)
        printf("Approximate Pattern Mathing: "
               "looking for %d pattern(s) in file %s w/ distance of %d\n",
               nb_patterns, filename, approx_factor);

    own_buf = read_input_file(filename, &own_n_bytes);
    if (own_buf == NULL)
    {
        own_n_bytes = 0;
    }

    /* Allocate the array of matches */
    n_matches = (int *)malloc(nb_patterns * sizeof(int));
    if (n_matches == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                nb_patterns * sizeof(int));
        return 1;
    }

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

    int max_pattern_length = longest_string_len(pattern, nb_patterns);

    for (i = 0; i < nb_patterns; i++)
    {
        n_matches[i] = 0;
    }

    // send data to people who need it asynchronically
    MPI_Request req;
    for (i = 0; i < comm_size; i++)
    {
        int other_start, other_end_data;
        fill_data_bounds(i, comm_size, nb_patterns, max_pattern_length, n_bytes, &other_start, NULL, &other_end_data, NULL, NULL);

        if (intersect(own_start, own_end, other_start, other_end_data))
        {
            int s = MAX2(other_start, own_start);
            int e = MIN2(other_end_data, own_end);
            int length = e - s;
            if (length < 0)
            {
                printf("ERROR: length is negative! (start: %d, end: %d, other_start: %d, other_end_data: %d, own_start: %d, own_end: %d)\n", s, e, other_start, other_end_data, own_start, own_end);
                return 1;
            }
            MPI_Isend(own_buf + s - own_start, length, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
        }
    }

    int start, end, end_data, starti, endi;
    fill_data_bounds(rank, comm_size, nb_patterns, max_pattern_length, n_bytes, &start, &end, &end_data, &starti, &endi);
    char *actual_data = (char *)malloc((end_data - start) * sizeof(char));
    if (actual_data == NULL)
    {
        fprintf(stderr, "Error: unable to allocate memory for %ldB\n",
                (end_data - start) * sizeof(int));
        return 1;
    }

    // fill the actual data by recieving from other processes
    for (i = 0; i < comm_size; i++)
    {
        int other_start = displs[i];
        int other_end_data = displs[i] + lengths[i];

        if (intersect(start, end_data, other_start, other_end_data))
        {
            int s = MAX2(other_start, start);
            int e = MIN2(other_end_data, end_data);
            int length = e - s;
            if (length < 0)
            {
                printf("ERROR: length is negative! (start: %d, end: %d, other_start: %d, other_end_data: %d, own_start: %d, own_end: %d)\n", s, e, other_start, other_end_data, own_start, own_end);
                return 1;
            }
            MPI_Recv(actual_data + s - start, length, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

#if APM_DEBUG
    printf("Rank %d: start: %d, end: %d, end_data: %d, starti: %d, endi: %d, nb_patterns: %d, n_bytes: %d, max_pattern_length: %d", rank, start, end, end_data, starti, endi, nb_patterns, n_bytes, max_pattern_length);
#endif

    buf = actual_data - start; // make the buffer start at "the beginning" of the data

    /* Check each pattern one by one */
#pragma omp parallel private(i, j) shared(buf, pattern, n_matches, n_bytes, nb_patterns, approx_factor, start, end, starti, endi, max_pattern_length) default(none)
    {
        int *column;
        column = (int *)malloc((max_pattern_length + 1) * sizeof(int));
        if (column == NULL)
        {
            printf("Error: unable to allocate memory for column (%ldB)\n",
                   (max_pattern_length + 1) * sizeof(int));
            exit(1);
        }
#pragma omp for schedule(static) collapse(2)
        /* Traverse the patterns */
        for (i = starti; i < endi; i++)
        {
            /* Traverse the input data up to the end of the file */
            for (j = start; j < end; j++)
            {
                int size_pattern = strlen(pattern[i]);
                int distance = 0;
                int size;

                size = size_pattern;
                if (n_bytes - j < size_pattern)
                {
                    size = n_bytes - j;
                }

                distance = levenshtein(pattern[i], &buf[j], size, column);

                if (distance <= approx_factor)
                {
#pragma omp atomic
                    n_matches[i]++;
                    // printf("Match found at position %d for pattern <%s> (distance: %d)\n",
                    //        j, pattern[i], distance);
                }
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
        printf("%s done in %lf s (size ; %d)\n\n", argv[0], duration, comm_size);

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
