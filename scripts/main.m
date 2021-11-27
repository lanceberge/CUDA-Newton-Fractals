function main(test, step)
    if (nargin < 1)
        error("run as main <test> [step]")
    elseif (nargin < 2)
        create_plot(test);
    else
        for i=0:99
            create_plot(test, i);
            pause;
        end
    end
end