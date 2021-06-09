#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <wchar.h>

int Sort(const char *name, int startNum)
{
	printf("--Sort_Start\n");
	
	int maxLen = 0, strNum = 0, neof = 1, n = 0;
	int f, ftmp;
	wchar_t *c = malloc(1);
	
	f = open(name, O_RDWR, 0777);
	if(f == -1)
		return -1;
	ftmp = open("ftmp", O_RDWR | O_CREAT | O_TRUNC, 0777);
	if(ftmp == -1)
		return -1;
	
	while(neof)
	{
		while((*c != '\n') && (neof))
		{
			neof = read(f, c, 1);
			n++;
		}
		if(n > maxLen) maxLen = n;
		n = 0;
		if((*c == '\n') && neof) strNum++;
		*c = 'a';
	}
	
	lseek(f, 0, SEEK_SET);
	char flags[strNum];
	int strLens[strNum];
	*c = 'a';
	
	for(int i = 0; i < strNum; i++)
	{
		flags[i] = 0;
	}
	for(int i = 0; i < startNum; i++)
	{
		flags[i] = 1;
	}
	
	for(int i = 0; i < strNum; i++)
	{
		while(*c != '\n')
		{
			read(f, c, 1);
			n++;
		}
		strLens[i] = n;
		n = 0;
		*c = 'a';
	}

	printf("strNum = %i\n", strNum);
	printf("maxLen = %i\n\n", maxLen);
	for(int i = 0; i < strNum; i++)
	{
		printf("strLens[i] = %i\n", strLens[i]);
	}
	
	wchar_t minStr[maxLen];
	char fl = 0;
	int m;
	
	for(int i = 0; i < strNum; i++)
	{
		n = 0;
		m = 0;
		for(int j = 0; (flags[j] != 0) && (j < strNum); j++)
			n += strLens[j];
		lseek(f, n, SEEK_SET);
		for(int j = 0; *c != '\n'; j++)
		{
			read(f, c, 1);
			minStr[j] = *c;
		}
		*c = 'a';
		
		n = 0;
		for(int l = 0; l < strNum; l++)
		{
			if(flags[l] != 0)
				continue;
			for(int j = 0; j < l; j++)
				n += strLens[j];
			lseek(f, n, SEEK_SET);
			for(int k = 0; *c != '\n'; k++)
			{
				read(f, c, 1);
				if((*c < minStr[k]) || ((*c == '\n') && (minStr[k] != '\n')))
				{
					lseek(f, n, SEEK_SET);
					*c = 'a';
					for(int j = 0; *c != '\n'; j++)
					{
						read(f, c, 1);
						minStr[j] = *c;
					}
					m = 1;
					*c = 'a';
					break;
				}
				else if((*c > minStr[k]) || ((*c != '\n') && (minStr[k] == '\n')))
				 	break;
				else if((*c == '\n') && (minStr[k] == '\n'))
				{
					m++;
					break;
				}
			}
			*c = 'a';
			n = 0;
		}
		
		printf("%i ", m);
		for(int l = 0; minStr[l] != '\n'; l++)
		{
			printf("%c", minStr[l]);
		}
		printf("\n");
		
		for(int l = 0; l < strNum; l++)
		{
			if(flags[l] != 0)
				continue;
			for(int j = 0; j < l; j++)
				n += strLens[j];
			lseek(f, n, SEEK_SET);
			for(int k = 0; *c != '\n'; k++)
			{
				read(f, c, 1);
				if((*c < minStr[k]) || ((*c == '\n') && (minStr[k] != '\n')))
					break;
				else if((*c > minStr[k]) || ((*c != '\n') && (minStr[k] == '\n')))
					break;
				else if((*c == '\n') && (minStr[k] == '\n'))
				{
					flags[l] = 1;
					break;
				}
			}
			*c = 'a';
			n = 0;
		}
		
		for(int l = 0; l < strNum; l++)
		{
			printf("%i ", flags[l]);
		}
		printf("\n");
		
		for(int l = 0; l < m; l++)
		{
			for(int j = 0; minStr[j] != '\n'; j++)
			{
				*c = minStr[j];
				write(ftmp, c, 1);
			}
			*c = '\n';
			write(ftmp, c, 1);
			*c = 'a';
		}
		
		for(int l = 0; l < strNum; l++)
			if(flags[l] == 0)
				fl = 1;
		
		if(fl == 0) 
			break;
		else fl = 0;
	}
	
	printf("--Sort_End\n\n");
	
	return ftmp;
}

int ReverseFile(const char *name)
{
	printf("--ReverseFile_Start\n");
	
	int f, ftmp, neof = 1;
	int strNum = 0, n = 0;
	wchar_t *c = malloc(1);
	
	f = open(name, O_RDWR, 0777);
	if(f == -1)
		return -1;
	ftmp = open("ftmp1", O_RDWR | O_CREAT | O_TRUNC, 0777);
	if(ftmp == -1)
		return -1;
	
	neof = read(f, c, 1);
	while(neof)
	{
		if(*c == '\n')
			strNum++;
		neof = read(f, c, 1);
	}
	
	int strLens[strNum];
	lseek(f, 0, SEEK_SET);
	*c = 'a';
	
	for(int i = 0; i < strNum; i++)
	{
		while(*c != '\n')
		{
			read(f, c, 1);
			n++;
		}
		strLens[i] = n;
		n = 0;
		*c = 'a';
	}
	
	printf("strNum = %i\n", strNum);
	for(int i = 0; i < strNum; i++)
	{
		printf("strLens[i] = %i\n", strLens[i]);
	}
	
	for(int i = 0; i < strNum; i++)
	{
		for(int j = 0; j < (strNum - i - 1); j++)
			n += strLens[j];
		lseek(f, n, SEEK_SET);
		while(*c != '\n')
		{
			read(f, c, 1);
			write(ftmp, c, 1);
		}
		n = 0;
		*c = 'a';
	}
	
	printf("--ReverseFile_End\n\n");
	
	return ftmp;
}

int main(int argc, char *argv[])
{
	int argR = 0, argN = 0, argM = 0, argO = 0, filesStart = 0, neof = 1;
	int fnew, fnewNum, ftmp, ftmp1, fcur, fo;
	char *c = malloc(1);
	*c = 'a';
	
	printf("--Params_Check_Start\n");
	
	for(int i = 1; i <= argc; i++)
	{
		if(*(argv[i]) == '-')
			switch(*(argv[i] + 1))
			{
				case 'r':
					argR = 1;
					break;
				case 'm':
					argM = 1;
					break;
				case 'o':
					argO = 1;
					i++;
					fnewNum = i;
					break;
			}
		else if(*(argv[i]) == '+')
			argN = atoi(argv[i] + 1) - 1;
		else 
		{
			filesStart = i;
			break;
		}
	}
	
	printf("argR = %i, argN = %i, argM = %i, argO = %i, filesStart = %i, fnewNum = %i\n", argR, argN, argM, argO, filesStart, fnewNum);
	
	printf("--Params_Check_End\n\n");
	
	fnew = open("fnew", O_RDWR | O_CREAT | O_TRUNC, 0777);
	if(fnew == -1)
	{
		printf("File error!");
		return -1;
	}
	
	for(int i = filesStart; i < argc; i++)
	{
		fcur = open(argv[i], O_RDWR, 0777);
		if(fcur == -1)
		{
			printf("File error!");
			return -1;
		}
		ftmp = Sort(argv[i], argN);
		lseek(ftmp, 0, SEEK_SET);
		
		if(argR)
		{
			for(int j = 0; j < argN; j++)
			{
				while(*c != '\n')
				{
					read(fcur, c, 1);
					write(fnew, c, 1);
				}
				*c = 'a';
			}
			ftmp1 = ReverseFile("ftmp");
			lseek(ftmp1, 0, SEEK_SET);
			neof = read(ftmp1, c, 1);
			while(neof)
			{
				write(fnew, c, 1);
				neof = read(ftmp1, c, 1);
			}
			*c = 'a';
			neof = 1;
			close(ftmp);
			close(ftmp1);
		}
		else
		{
			for(int j = 0; j < argN; j++)
			{
				while(*c != '\n')
				{
					read(fcur, c, 1);
					write(fnew, c, 1);
				}
				*c = 'a';
			}
			neof = read(ftmp, c, 1);
			while(neof)
			{
				write(fnew, c, 1);
				neof = read(ftmp, c, 1);
			}
			*c = 'a';
			neof = 1;
			close(ftmp);
		}
	}
	
	close(fcur);
	unlink("ftmp");
	unlink("ftmp1");
	
	if(argO)
	{
		fo = open(argv[fnewNum], O_RDWR | O_CREAT | O_TRUNC, 0777);
		if(fo == -1)
		{
			printf("File error!");
			return -1;
		}
		lseek(fnew, 0, SEEK_SET);
		neof = read(fnew, c, 1);
		while(neof)
		{
			write(fo, c, 1);
			neof = read(fnew, c, 1);
		}
		close(fo);
	}
	else
	{
		lseek(fnew, 0, SEEK_SET);
		neof = read(fnew, c, 1);
		while(neof)
		{
			write(1, c, 1);
			neof = read(fnew, c, 1);
		}
	}
	
	close(fnew);
	unlink("fnew");
	
	return 0;
}