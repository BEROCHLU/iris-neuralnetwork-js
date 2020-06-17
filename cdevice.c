#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define SIZE            8 //string size
#define DESIRED_ERROR   0.001 //allowable error
#define THRESHOLD       20000 //epoch limit
#define IN_NODE         4 //str,float,float,float
#define HID_NODE        5 //IN_NODE+1
#define OUT_NODE        1
#define ETA             0.5 //learning coefficient
#define sigmoid(x)      (1.0 / (1.0 + exp(-x)))
#define dsigmoid(x)     ((x) * (1.0 - (x)))
#define dfmax(x)        ((x) > 0 ? 1.0 : 0)
#define ACTIVE_MODE     0 //0: sigmoid, 1:ReLU

void FindHiddenOutput(int);
void PrintResult(void);
void PrintVW(void);
double FrandFix(void);

int q = 0, days;
char date[SIZE][12];
double DOWdiv = 0, FXdiv = 0, N225div = 0;
double x[SIZE][IN_NODE], t[SIZE][OUT_NODE];
double v[HID_NODE][IN_NODE], w[OUT_NODE][HID_NODE], hid[HID_NODE], out[OUT_NODE];

int main()
{
    int i, j, k, p;
    double Error = DBL_MAX;
    double delta_out[OUT_NODE], delta_hid[HID_NODE];

    FILE *fp;
    clock_t start, end;
    time_t timer;

    if ((fp = fopen("./csv/cell-automaton30-ca.csv", "r")) == NULL) {
        printf("The file doesn't exist!\n"); exit(1);
    }

    for (i = 0; EOF != fscanf(fp, "%[^,],%lf,%lf,%lf,%lf", date[i], &x[i][0], &x[i][1], &x[i][2], &t[i][0]); i++) {
        x[i][3] = FrandFix();//配列最後にバイアス
    }

    days = i - 1;

    fclose(fp);

    srand((unsigned int)time(NULL));    /**現在時刻を元に種を生成*/

    for (i = 0; i < HID_NODE; i++)      /* 中間層の結合荷重を初期化 */
        for (j = 0; j < IN_NODE; j++)
            v[i][j] = FrandFix();

    for (i = 0; i < OUT_NODE; i++) /* 出力層の結合荷重の初期化 */
        for (j = 0; j < HID_NODE; j++)
            w[i][j] = FrandFix();

    time(&timer);
    printf("%s", ctime(&timer));

    start = clock();

    while (DESIRED_ERROR < Error) {
        q++; Error = 0;

        for (p = 0; p < days; p++) {
            FindHiddenOutput(p);

            for (k = 0; k < OUT_NODE; k++) {
                Error += 0.5 * pow(t[p][k] - out[k], 2.0);                  //誤差を日数分加算する
                // Δw
                delta_out[k] = (t[p][k] - out[k]) * out[k] * (1 - out[k]);  //δ=(t-o)*f'(net); net=Σwo; δo/δnet=f'(net);
            }

            for (k = 0; k < OUT_NODE; k++) {// Δw
                for (j = 0; j < HID_NODE; j++) {
                    w[k][j] += ETA * delta_out[k] * hid[j];//Δw=ηδH
                }
            }

            for (i = 0; i < HID_NODE; i++) {// Δv
                delta_hid[i] = 0;

                for (k = 0; k < OUT_NODE; k++) {
                    delta_hid[i] += delta_out[k] * w[k][i];//Σδw
                }

                if (ACTIVE_MODE == 0){
                    delta_hid[i] = dsigmoid(hid[i]) * delta_hid[i];//H(1-H)*Σδw
                } else if (ACTIVE_MODE == 1){
                    delta_hid[i] = dfmax(hid[i]) * delta_hid[i];//H(1-H)*Σδw
                }
            }

            for (i = 0; i < HID_NODE; i++) {// Δv
                for (j = 0; j < IN_NODE; j++) {
                    v[i][j] += ETA * delta_hid[i] * x[p][j];//Δu=ηH(1-H)XΣδw
                }
            }
            q = q + 0;// debug
        }

        if (q % 100 == 0)
            printf("%6d: %f\n", q, Error);
        if (THRESHOLD < q)
            break;
    }

    end = clock();

    /* 学習結果を表示 */
    PrintResult();

    printf("Time %.2lfsec.\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}


void FindHiddenOutput(int p)
{
    int i, j; double neth, neto;

    for (i = 0; i < HID_NODE; i++) {
        neth = 0;

        for (j = 0; j < IN_NODE; j++)
            neth += v[i][j] * x[p][j];
        
        if (ACTIVE_MODE == 0){
            hid[i] = sigmoid(neth);
        } else if (ACTIVE_MODE == 1){
            hid[i] = fmax(0, neth);
        }
    }

    if (ACTIVE_MODE == 0){
        hid[HID_NODE - 1] = FrandFix(); //add bias in HIDDEN NODE
    } else if (ACTIVE_MODE == 1){
        hid[HID_NODE - 1] = -1; //add bias in HIDDEN NODE
    }
    

    for (i = 0; i < OUT_NODE; i++) {
        neto = 0;

        for (j = 0; j < HID_NODE; j++)
            neto += w[i][j] * hid[j];

        out[i] = sigmoid(neto);
    }
}

/* Print out the final result */
void PrintResult(void)
{
    int i;

    for (i = 0; i < days; i++) {
        FindHiddenOutput(i);

        printf("%s\t%6.3lf True %6.3lf", date[i], out[0], t[i][0]);
    }
    printf("\n");
}

void PrintVW(void)
{
    int i, j;

    for (i = 0; i < HID_NODE; i++) {/* 中間層の結合荷重 */
        printf("v = ");
        for (j = 0; j < IN_NODE; j++)
            printf("%5lf ", v[i][j]);
        printf("\n");
    }

    for (i = 0; i < OUT_NODE; i++) {/* 出力層の結合荷重 */
        printf("w = ");
        for (j = 0; j < HID_NODE; j++)
            printf("%5lf ", w[i][j]);
        printf("\n");
    }
}

/**fix same seed issue of random number*/
double FrandFix(void)
{
    int i;
    double drand;

    /**乱数を複数回生成して最後の値を使用する*/
    for (i = 0; i < 101; i++)
        drand = rand() % 10000 / 10001.0;

    return drand;
}
