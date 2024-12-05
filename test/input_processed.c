#include <stdint.h>

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
    int deref = PTR_DEREF((ptr), ((myPair->first) * 1.5));

}