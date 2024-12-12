#include <stdlib.h>
#define PTR_DEREF(p,q) (*(p) + q)

typedef struct {
    int first;
    int second;
} Pair;

int main() {
    Pair* myPair = (Pair*)malloc(sizeof(Pair));

    myPair->first = 42;
    myPair->second = 84;
    

    int val = 123;
    int* ptr = &val;
    int deref1;
    deref1 = PTR_DEREF((ptr), (20));
    double deref2 = PTR_DEREF((ptr), ((myPair->first) * 1.5));
}