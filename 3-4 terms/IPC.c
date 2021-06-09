#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <signal.h>

pid_t chPid1, chPid2;
key_t key;
int f;
int semid, shmid;
int check, restart;
char *shmaddr;
struct shmid_ds shm;
struct sembuf sops;

void PrintField(void)
{
	for(int i = 0; i < 3; i++)
	{
		for(int j = 0; j < 3; j++)
			printf("%c", shmaddr[i*3 + j]);
		printf("\n");
	}
	printf("\n");
}

void Exit(int num)
{
	if(num == 1)
	{
		shmaddr[0] = 0;
		shmdt(shmaddr);
		semop(semid, &sops, 1);
		exit(0);
	}
	else if(num == 2)
	{
		shmaddr[0] = 0;
		shmdt(shmaddr);
		sops.sem_op = 2;
		semop(semid, &sops, 1);
		exit(0);
	}
	else
	{
		shmdt(shmaddr);
		shmctl(shmid, IPC_RMID, &shm);
		semctl(semid, IPC_RMID, (int) 0);
		close(f);
		unlink("shm");
	}
}

void Clear(int num)
{
	if((chPid1 = fork()) == -1)
	{
		perror("fork");
		exit(-1);
	}
	if(!chPid1)
	{
		execlp("clear", "clear", NULL);
		perror("clear");
		Exit(num);
	}
	wait(NULL);
}

int main(void)
{
	sops.sem_num = 0;
	sops.sem_flg = 0;
	
	if((f = open("shm", O_RDWR | O_CREAT, 0777)) == -1)
	{
		perror("open");
		return -1;
	}
	
	key = ftok("shm", 'a');
	
	if((semid = semget(key, 1, 0666 | IPC_CREAT)) == -1)
	{
		perror("semget");
		return -1;
	}
	
	if((shmid = shmget(key, 10, 0666 | IPC_CREAT)) == -1)
	{
		perror("shmget");
		return -1;
	}
	shmaddr = shmat(shmid, NULL, 0);
	
	do
	{
		Clear(0);
		
		restart = -1;
		semctl(semid, 0, SETVAL, (int) 1);
		for(int i = 0; i < 9; i++)
			shmaddr[i] = '_';
		shmaddr[9] = 0;

		if((chPid1 = fork()) == -1)
		{
			perror("fork");
			return -1;
		}

		if(chPid1)
		{
			if((chPid2 = fork()) == -1)
			{
				perror("fork");
				kill(chPid1, SIGKILL);
				wait(NULL);
				return -1;
			}

			if(chPid2)
			{
				//father
				wait(NULL);
				wait(NULL);
			}
			else
			{
				//son2
				int field = -1;

				key = ftok("shm", 'a');

				if((semid = semget(key, 1, 0666 | IPC_CREAT)) == -1)
				{
					perror("semget");
					exit(-1);
				}

				if((shmid = shmget(key, 10, 0666 | IPC_CREAT)) == -1)
				{
					perror("shmget");
					exit(-1);
				}  
				shmaddr = shmat(shmid, NULL, 0);

				int stats[8];

				sops.sem_op = 0;
				semop(semid, &sops, 1);
				while(1)
				{
					//telo
					if(shmaddr[0] == 0)
						Exit(2);
					
					Clear(2);

					printf("--2nd player's turn\n\n");
					PrintField();

					while((field < 0) || (field > 9))
					{
						printf("Enter number > 0 and < 10 or 0 to finish: ");
						scanf("%i", &field);
						if((field > 0) && (field < 10) && shmaddr[field - 1] != '_')
						{
							printf("This field is taken try again\n");
							field = -1;
						}
					}

					if(field == 0)
					{
						Clear(2);
						printf("Game was finished\n\n");
						Exit(2);
					}

					shmaddr[field - 1] = 'o';

					stats[(field - 1) % 3]++;
					stats[((field - 1) / 3) + 3]++;
					if((field == 1) || (field == 9))
						stats[6]++;
					else if((field == 3) || (field == 7))
						stats[7]++;
					else if(field == 5)
					{
						stats[6]++;
						stats[7]++;
					}
					shmaddr[9]++;
					for(int i = 0; i < 8; i++)
					{
						if(stats[i] == 3)
						{
							Clear(2);
							
							printf("2nd player WINS!!!\n\n");
							PrintField();

							Exit(2);
						}
					}
					if(shmaddr[9] == 9)
					{
						Clear(2);
						
						printf("DRAW!!!\n\n");
						PrintField();
						
						Exit(2);
					}

					field = -1;

					sops.sem_op = 2;
					semop(semid, &sops, 1);
					sops.sem_op = 0;
					semop(semid, &sops, 1);
				}
			}
		}
		else
		{
			//son1
			int field = -1;

			key = ftok("shm", 'a');

			if((semid = semget(key, 1, 0666 | IPC_CREAT)) == -1)
			{
				perror("semget");
				exit(-1);
			}

			if((shmid = shmget(key, 10, 0666 | IPC_CREAT)) == -1)
			{
				perror("shmget");
				exit(-1);
			}	   
			shmaddr = shmat(shmid, NULL, 0);

			int stats[8];

			sops.sem_op = -1;
			while(1)
			{
				//telo
				if(shmaddr[0] == 0)
					Exit(2);
				
				Clear(1);

				printf("--1st player's turn\n\n");
				PrintField();

				while((field < 0) || (field > 9))
				{
					printf("Enter number > 0 and < 10 or 0 to finish: ");
					scanf("%i", &field);
	
					if((field > 0) && (field < 10) && shmaddr[field - 1] != '_')
					{
						printf("This field is taken try again\n");
						field = -1;
					}
				}

				if(field == 0)
				{
					Clear(1);
					printf("Game was finished\n\n");
					Exit(1);
				}

				shmaddr[field - 1] = 'x';
				stats[(field - 1) % 3]++;
				stats[((field - 1) / 3) + 3]++;
				if((field == 1) || (field == 9))
					stats[6]++;
				else if((field == 3) || (field == 7))
					stats[7]++;
				else if(field == 5)
				{
					stats[6]++;
					stats[7]++;
				}
				shmaddr[9]++;
				
				for(int i = 0; i < 8; i++)
				{
					if(stats[i] == 3)
					{
						Clear(1);
						
						printf("1nd player WINS!!!\n\n");
						PrintField();
						
						Exit(1);
					}
				}
				if(shmaddr[9] == 9)
				{
					Clear(1);
					
					printf("DRAW!!!\n\n");
					PrintField();

					Exit(1);
				}

				field = -1;

				semop(semid, &sops, 1);
				semop(semid, &sops, 1);
			}
		}
		while((restart != 0) && (restart != 1))
		{
			printf("Do you want to play again? [0 - No / 1 - Yes]\n");
			scanf("%i", &restart);
		}
			
	} while(restart);
	
	Clear(0);
	
	Exit(0);
	
	return 0;
}