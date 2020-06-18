#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define SIZE            8       //char size
#define DESIRED_ERROR   0.001   //threshold of loss error
#define THRESHOLD       20000   //threshold of epoch
#define IN_NODE         4       //str,float,float,float
#define HID_NODE        5       //IN_NODE+1
#define OUT_NODE        1
#define ETA             0.5     //learning coefficient
#define sigmoid(x)      (1.0 / (1.0 + exp(-x)))
#define dsigmoid(x)     ((x) * (1.0 - (x)))
#define dfmax(x)        ((x) > 0 ? 1.0 : 0)

void calcHiddenOutput(int);
void printResult(void);
double frandFix(void); //fix same seed issue of random number

int DATA_LEN;
int ACTIVE_MODE = 0; //0: sigmoid, 1:ReLU
char date[SIZE][16];
double x[SIZE][IN_NODE], t[SIZE][OUT_NODE];
double v[HID_NODE][IN_NODE], w[OUT_NODE][HID_NODE];
double hid[HID_NODE], out[OUT_NODE];

int main(int argc, char *argv[])
{
    int i, j, k, n;
    int epoch = 0;
    double loss_error = DBL_MAX;
    double delta_out[OUT_NODE], delta_hid[HID_NODE];

    FILE *fp;
    clock_t start, end;
    time_t timer;

    if ((fp = fopen("./csv/cell-automaton30.csv", "r")) == NULL) {
        printf("The file doesn't exist!\n"); exit(1);
    }

    for (i = 0; EOF != fscanf(fp, "%[^,],%lf,%lf,%lf,%lf", date[i], &x[i][0], &x[i][1], &x[i][2], &t[i][0]); i++) {
        x[i][3] = frandFix();//add bias in INPUT NODE
    }

    DATA_LEN = i - 1;

    fclose(fp);
    srand((unsigned int)time(NULL));    //generate a seed based on the current time
    //argcの個数で切り分けることで引数あるなしの処理を分けられる
    if (1 < argc) {
        if (strcmp(argv[1], "0") == 0) {
            ACTIVE_MODE = 0;
        } else if(strcmp(argv[1], "1") == 0) {
            ACTIVE_MODE = 1;
        }
    }

    for (i = 0; i < HID_NODE; i++)      // initialize input-hidden weight
        for (j = 0; j < IN_NODE; j++)
            v[i][j] = frandFix();

    for (i = 0; i < OUT_NODE; i++) // initialize hidden-output weight
        for (j = 0; j < HID_NODE; j++)
            w[i][j] = frandFix();

    time(&timer);
    printf("%s", ctime(&timer));

    start = clock();

    while (DESIRED_ERROR < loss_error) {
        epoch++;
        loss_error = 0;

        for (n = 0; n < DATA_LEN; n++) {
            calcHiddenOutput(n);

            for (k = 0; k < OUT_NODE; k++) {
                loss_error += 0.5 * pow(t[n][k] - out[k], 2.0);             //least squares method
                // Δw
                delta_out[k] = (t[n][k] - out[k]) * out[k] * (1 - out[k]);  //δ=(t-o)*f'(net); net=Σwo; δo/δnet=f'(net);
            }

            for (k = 0; k < OUT_NODE; k++) {                // Δw
                for (j = 0; j < HID_NODE; j++) {
                    w[k][j] += ETA * delta_out[k] * hid[j]; //Δw=ηδH
                }
            }

            for (i = 0; i < HID_NODE; i++) {// Δv
                delta_hid[i] = 0;

                for (k = 0; k < OUT_NODE; k++) {
                    delta_hid[i] += delta_out[k] * w[k][i];//Σδw
                }

                if (ACTIVE_MODE == 0)
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i]; //H(1-H)*Σδw
                else if (ACTIVE_MODE == 1)
                    delta_hid[i] = dfmax(hid[i]) * delta_hid[i];    //H(1-H)*Σδw
            }

            for (i = 0; i < HID_NODE; i++) {                    // Δv
                for (j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[n][j];    //Δu=ηH(1-H)XΣδw
                }
            }
        }

        if (epoch % 100 == 0)
            printf("%6d: %f\n", epoch, loss_error);
        if (THRESHOLD < epoch)
            break;
    }

    end = clock();

    printResult();
    printf("Time %.2lfsec.\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}


void calcHiddenOutput(int n)
{
    int i, j; double neth, neto;

    for (i = 0; i < HID_NODE; i++) {
        neth = 0;

        for (j = 0; j < IN_NODE; j++)
            neth += v[i][j] * x[n][j];

        if (ACTIVE_MODE == 0)
            hid[i] = sigmoid(neth);
        else if (ACTIVE_MODE == 1)
            hid[i] = fmax(0, neth);
    }

    if (ACTIVE_MODE == 0)
        hid[HID_NODE - 1] = frandFix(); //add bias in HIDDEN NODE
    else if (ACTIVE_MODE == 1)
        hid[HID_NODE - 1] = -1;         //add bias in HIDDEN NODE


    for (i = 0; i < OUT_NODE; i++) {
        neto = 0;

        for (j = 0; j < HID_NODE; j++)
            neto += w[i][j] * hid[j];

        out[i] = sigmoid(neto);
    }
}

void printResult(void)
{
    int i;

    for (i = 0; i < DATA_LEN; i++) {
        calcHiddenOutput(i);//Predict with test data based on training data
        printf("%s\t%6.3lf True %6.3lf", date[i], out[0], t[i][0]);
    }
    printf("\n");
}

double frandFix(void)
{
    int i;
    double drand;

    //Generate random number multiple times and use last value
    for (i = 0; i < 101; i++)
        drand = rand() % 10000 / 10001.0;

    return drand;
}
