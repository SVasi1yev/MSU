#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int Shell(char *command);

char* EnterString(void)
{
	char *s,*p;
	const int n = 5;
	s = (char*) malloc(n * sizeof(char));
	p = s;
	int count = n;
	for(;;)
	{
		for(int i = 0; i < n; i++,p++)
		{
			if((*p = getchar()) == '\n') break;
		}
		if(*p == '\n') 
		{
			*p = '\0';
			break;	
		}
		s = (char *) realloc (s,(count + n) * sizeof(char));
		count += n;
	}
	
	return s;
}

int Conveyor(char *conv, int ins1)
{
	printf("--Conveyor_Start\n");
	
	int comCnt = 1;
	int status;
	char *inFile = NULL;
	char *outFile = NULL;
	int outFileMode = 0;
	pid_t pid = -2;
	
	for(int i = 0; conv[i] != '\0'; i++)
	{
		if(conv[i] == '(')
		{
			//обработка скобок
			int skobCnt = 1;
			int l0 = i;
			int l1 = l0;
			char *newShellCom = NULL;
			while(conv[i] != '\0')
			{
				i++;
				if(conv[i] == '(')
					skobCnt++;
				else if(conv[i] == ')')
					skobCnt--;
				if(skobCnt == 0)
				{
					l1 = i;
					newShellCom = malloc(l1 - l0);
					for(int j = 1; j < (l1 - l0); j++)
						newShellCom[j - 1] = conv[l0 + j];
					newShellCom[l1 - l0 - 1] = '\0';
					
					printf("--Conveyor_End\n");
					return Shell(newShellCom);
				}
			}
			printf("--Conveyor_End\n");
			return -1;
		}
		else if(conv[i] == '|')
			comCnt++;
		else if(conv[i] == '<')
		{
			int j;
			inFile = malloc(255);
			i++;
			for(j = 0;(conv[i] != ' ') && (conv[i] != '\0'); j++, i++)
				inFile[j] = conv[i];
			inFile[j] = '\0';
			i--;
		}
		else if(conv[i] == '>')
		{
			if(conv[i + 1] == '>')
			{
				int j;
				outFileMode = O_APPEND;
				outFile = malloc(255);
				i += 2;
				for(j = 0;(conv[i] != ' ') && (conv[i] != '\0'); j++, i++)
					outFile[j] = conv[i];
				inFile[j] = '\0';
				i--;
			}
			else
			{
				int j;
				outFileMode = O_TRUNC;
				outFile = malloc(255);
				i++;
				for(j = 0;(conv[i] != ' ') && (conv[i] != '\0'); j++, i++)
					outFile[j] = conv[i];
				inFile[j] = '\0';
				i--;
			}
		}
	}
	
	//формирование команд
	char** commands[comCnt];
	for(int i = 0; i < comCnt; i++)
		commands[i] = malloc(sizeof(char*));
	int k = 0, n1 = 0, n2 = 0, n3 = 0;
	char *str = malloc(1);
	
	while(1)
	{
		if(conv[k] != ' ')
		{
			if(conv[k] == '|')
			{
				commands[n1][n2] = NULL;
				n2 = 0;
				n1++;
			}
			else if((conv[k] == '\0') || (conv[k] == '<') || (conv[k] == '>'))
			{
				commands[n1][n2] = NULL;
				n2 = 0;
				break;
			}
			else
			{
				while((conv[k] != ' ') && (conv[k] != '|') && (conv[k] != '\0') && (conv[k] != '<') && (conv[k] != '>'))
				{
					str = realloc(str, (n1 + 1));
					str[n3] = conv[k];
					n3++;
					k++;
				}
				str[n3] = '\0';
				commands[n1] = realloc(commands[n1], (n2 + 1) * sizeof(char*));
				commands[n1][n2] = str;
				n3 = 0;
				str = malloc(1);
				n2++;
				k--;
			}
		}
		
		k++;
	}
	
	//формирование каналов
	int* pipes[comCnt - 1];
	for(int i = 0; i < (comCnt - 1); i++)
		pipes[i] = malloc(2 * sizeof(int));
	
	for(int i = 0; i < (comCnt - 1); i++)
		pipe(pipes[i]);
		
	
	//формирование файлов для перенаправления
	int inF = 0, outF = 1;
	
	if(inFile != NULL)
	{
		inF = open(inFile, O_RDONLY, 0777);
		if(inF == -1)
		{
			perror("open");
			printf("--Conveyor_End\n");
			return -1;
		}
	}
	
	if(outFile != NULL)
	{
		outF = open(outFile, O_CREAT | O_WRONLY | outFileMode, 0777);
		if(inF == -1)
		{
			perror("open");
			printf("--Conveyor_End\n");
			return -1;
		}
	}
	
	
	//конвейер
	if(ins1 == 1)
	{
		for(int i = 0; i < comCnt; i++)
		{
			if(i == 0)
			{
				if(inF != 0)
					dup2(0, inF);
					
				dup2(1, pipes[0][1]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					close(pipes[0][1]);
				if(!pid)
				{
					execvp(commands[i][0], commands[i]);
					close(pipes[0][1]);
					exit(-1);
				}
				 
			}
			else if(i == (comCnt - 1))
			{
				if(outF != 1)
					dup2(1, outF);
					
				dup2(0, pipes[comCnt - 2][0]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					return -1;
				if(!pid)
				{
					execvp(commands[i][0], commands[i]);
					exit(-1);
				}
			}
			else
			{
				dup2(1, pipes[i][1]);
				dup2(0, pipes[i - 1][0]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					close(pipes[i][1]);
				if(!pid)
				{
					execvp(commands[i][0], commands[i]);
					close(pipes[i][1]);
					exit(-1);
				}
			}
		}
	}
	//конвейер в фоновом режиме
	else
	{
		if(inF == 0)
		{
			int inF = open("/dev/null", O_RDWR, 0777);
			if(inF == -1)
			{
				perror("open");
				printf("--Conveyor_End\n");
				return -1;
			}
		}
		
		if(outF == 1)
		{
			int inF = open("/dev/null", O_RDWR, 0777);
			if(inF == -1)
			{
				perror("open");
				printf("--Conveyor_End\n");
				return -1;
			}
		}
		
		for(int i = 0; i < comCnt; i++)
		{
			if(i == 0)
			{
				dup2(0, inF);
				dup2(1, pipes[0][1]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					close(pipes[0][1]);
				if(!pid)
				{
					signal(SIGINT, SIG_IGN);
					execvp(commands[i][0], commands[i]);
					close(pipes[0][1]);
					exit(-1);
				}
				 
			}
			else if(i == (comCnt - 1))
			{
				dup2(1, outF);
				dup2(0, pipes[comCnt - 2][0]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					return -1;
				if(!pid)
				{
					signal(SIGINT, SIG_IGN);
					execvp(commands[i][0], commands[i]);
					exit(-1);
				}
			}
			else
			{
				dup2(1, pipes[i][1]);
				dup2(0, pipes[i - 1][0]);
				for(int j = 0; j < (comCnt - 1); j++)
				{
					if(j != i)
					close(pipes[j][1]);
				}
				
				if((pid = fork()) == -1)
					close(pipes[i][1]);
				if(!pid)
				{
					signal(SIGINT, SIG_IGN);
					execvp(commands[i][0], commands[i]);
					close(pipes[i][1]);
					exit(-1);
				}
			}
		}
	}
	for(int i = 0; i < comCnt; i++)
	{
		for(int j = 0; commands[i][j] != NULL; j++)
		{
			free(commands[i][j]);
		}
	}
	waitpid(pid, &status, 0);
	for(int i = 0; i < (comCnt - 1); i++)
		wait(NULL);
	
	if(WIFEXITED(status) && !WEXITSTATUS(status))
	{
		printf("--Conveyor_End\n");
		return 0;
	}
	else
	{
		printf("--Conveyor_End\n");
		return -1;
	}
	
	printf("--Conveyor_End\n");
	
	return 0;
}

int CondComm(char *condComm, int ins1)
{
	printf("--CondComm_Start\n");
	
	char *conv;
	char c = 'a';
	int m0 = -2, m1 = -2;
	int ins2 = 0;
	int success = 1;
	
	while(1)
	{
		m1 += 2;
		m0 = m1;
		ins2 = 0;
		success = 1;
		
		while(1)
		{
			if((condComm[m1] == '|') && (condComm[m1 + 1] == '|'))
			{
				ins2 = 1;
				break;
			}
			else if(condComm[m1] == '&')
			{
				ins2 = 2;
				break;
			}
			else if(condComm[m1] == '\0')
			{
				ins2 = 3;
				break;
			}
			m1++;
		}
		
		printf("ins2 = %i\n", ins2);
		
		conv = malloc(m1 - m0 + 1);
		for(int i = 0; i < (m1 - m0); i++)
			conv[i] = condComm[m0 + i];
		conv[m1 - m0] = '\0';
		
		success = Conveyor(conv, ins1);
		printf("success = %i\n", success);
		
		if(ins2 == 3)
		{
			free(conv);
			break;
		}
		else if((ins2 == 1) && (success == 0))
		{
			free(conv);
			break;
		}
		else if((ins2 == 2) && (success == -1))
		{
			free(conv);
			break;
		}
	}
	
	printf("--CondComm_End\n");
	
	return 0;
}

int Shell(char *command)
{
	printf("--Shell_Start\n");
	
	pid_t pid;
	char *condComm;
	int k0 = -1, k1 = -1;
	int fl1 = 0;
	int ins1 = 0;
	int status;
	
	while(1)
	{
		k0 = ++k1;
		ins1 = 0;
		
		while(1)
		{
			if(command[k1] == ';')
			{
				ins1 = 1;
				break;
			}
			else if(command[k1] == '\0')
			{
				ins1 = 1;
				fl1 = 1;
				break;
			}
			else if((command[k1] == '&') && (command[k1 + 1] != '&'))
			{
				ins1 = 2;
				break;
			}
			k1++;
		}
		
		printf("ins1 = %i\n", ins1);
		
		if((k1 - k0) == 0)
			break;
		
		if(ins1 == 1)
		{
			//последовательное выполнение
			condComm = malloc(k1 - k0 + 1);
			for(int i = 0; i < (k1 - k0); i++)
				condComm[i] = command[k0 + i];
			condComm[k1 - k0] = '\0';
			printf("%s\n", condComm);
			
			CondComm(condComm, ins1);
			
			free(condComm);
		}
		else if(ins1 == 2)
		{
			//выполнение в фоновом режиме
			if((pid = fork()) == -1)
			{
				perror("fork");
				continue;
			}
			if(!pid)
			{
				if((pid = fork()) == -1)
				{
					perror("fork");
					exit(-1);
				}
				if(!pid)
				{
					int f = open("/dev/null", O_RDWR, 0777);
					if(f == -1)
					{
						perror("open");
						exit(-1);
					}
					dup2(0, f);
					dup2(1, f);
					signal(SIGINT, SIG_IGN);
					condComm = malloc(k1 - k0 + 1);
					for(int i = 0; i <= (k1 - k0); i++)
						condComm[i] = command[k0 + i];
					condComm[k1 - k0] = '\0';
					CondComm(condComm, ins1);
					free(condComm);
					exit(0);
				}
				exit(0);
			}
			wait(&status);
//			if(status) 
//				return -1;
		}
		if(fl1)
			break;
	}
	
	printf("--Shell_End\n");
	
	return 0;
}

int main(void)
{
	char *command = NULL;
	while(1)
	{
		printf("Shell: ");
		command = EnterString();
		Shell(command);
	}
	
	return 0;
}
